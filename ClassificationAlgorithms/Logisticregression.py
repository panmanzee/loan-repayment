
import os
import sys
import time

import numpy as np

_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from data_classification import StandardScalerScratch           
from metrics_classification import average_precision, best_threshold, logloss  


def sigmoid(F):
    """Stable 1 / (1 + exp(-F))."""
    F = np.clip(F, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-F))


def sqrt_scale_pos_weight(y):
    """
    The class-imbalance weight :
        raw_weight       = (n_rows - n_positive) / n_positive
        scale_pos_weight = sqrt(raw_weight)
    """
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return 1.0
    return float(np.sqrt((len(y) - n_pos) / n_pos))


# -----------------------------------------------------------------------------
# The model
# -----------------------------------------------------------------------------
class LogisticRegressionScratch:

    def __init__(self, solver="gd", learning_rate=0.5, n_iterations=2000, reg_lambda=0.0, tol=1e-6,
                 standardize=True, class_weight=None, scale_pos_weight="auto", store_history=True):
        if solver not in ("gd", "newton"):
            raise ValueError('solver must be "gd" or "newton"')
        if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() != "auto":
            raise ValueError('scale_pos_weight must be "auto" or a number')
        self.solver = solver
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.reg_lambda = float(reg_lambda)
        self.tol = float(tol)
        self.standardize = bool(standardize)
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.store_history = bool(store_history)

    # -- internal helpers -----------------------------------------------------
    @staticmethod
    def _add_bias(X):
        return np.c_[np.ones(X.shape[0]), X]

    def _objective(self, Xb, y, w, theta):
        p = sigmoid(Xb @ theta)
        ll = -np.sum(w * (y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))))
        return float(ll / len(y) + self.reg_lambda * np.sum(theta[1:] ** 2) / (2 * len(y)))

    def _gradient(self, Xb, y, w, theta):
        n = len(y)
        p = sigmoid(Xb @ theta)
        grad = Xb.T @ (w * (p - y)) / n
        grad[1:] += self.reg_lambda * theta[1:] / n
        return grad, p

    def _hessian(self, Xb, w, p):
        n = Xb.shape[0]
        W = w * p * (1 - p)
        H = (Xb * W[:, None]).T @ Xb / n
        H[1:, 1:] += self.reg_lambda / n * np.eye(H.shape[0] - 1)
        return H

    def _sample_weights(self, y):
        if isinstance(self.scale_pos_weight, str) and self.scale_pos_weight.lower() == "auto":
            self.scale_pos_weight_ = sqrt_scale_pos_weight(y)
        else:
            self.scale_pos_weight_ = float(self.scale_pos_weight)
        w = np.where(y == 1, self.scale_pos_weight_, 1.0).astype(float)
        if self.class_weight is not None:
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            if self.class_weight == "balanced" and n0 and n1:
                w = w * np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
            else:
                raise ValueError('class_weight must be None or "balanced"')
        return w

    # -- optimisation core (one loop for both solvers) -------------------------
    def _run(self, X, y, w, eval_sets=None, early_stopping_rounds=None, eval_metric="logloss",
             verbose=False):
        Xb = self._add_bias(X)
        theta = np.zeros(Xb.shape[1])
        self.loss_history_, self.train_loss, self.theta_history_ = [], [], []
        self.converged_, self.gradient_norm_ = False, float("nan")
        self.iterations_run_, self.early_stopping_used_ = 0, False
        self.best_iteration_ = self.n_iterations
        self.best_score_ = None
        curve, best_score, best_theta = [], None, None
        grad = np.zeros(Xb.shape[1])

        for it in range(self.n_iterations):
            grad, p = self._gradient(Xb, y, w, theta)
            self.loss_history_.append(self._objective(Xb, y, w, theta))
            self.train_loss.append(logloss(y, p))

    
            if eval_sets:
                Xe, ye = eval_sets[0]
                p_va = sigmoid(self._add_bias(Xe) @ theta)
                score = float(logloss(ye, p_va) if eval_metric == "logloss"
                              else -average_precision(ye, p_va))
                curve.append(score)
                if best_score is None or score < best_score - 1e-12:
                    best_score, self.best_iteration_, best_theta = score, it + 1, theta.copy()
                elif (early_stopping_rounds is not None
                      and (it + 1) - self.best_iteration_ >= early_stopping_rounds):
                    self.early_stopping_used_ = True
                    break

            gnorm = float(np.linalg.norm(grad))
            self.gradient_norm_ = gnorm
            if gnorm < self.tol:                      
                self.converged_ = True
                break

            if self.solver == "gd":
                theta = theta - self.learning_rate * grad
            else:
                H = self._hessian(Xb, w, p)
                try:
                    theta = theta - np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:         
                    theta = theta - np.linalg.lstsq(H, grad, rcond=None)[0]
            self.iterations_run_ = it + 1              
            if self.store_history:
                self.theta_history_.append(theta.copy())   
            if verbose and (self.solver == "newton" or (it + 1) % 100 == 0):
                print(f"  {self.solver} iter {it + 1:>4} | loss {self.loss_history_[-1]:.6f} "
                      f"| |grad| {gnorm:.2e}")

        self.evals_result_ = {"validation": {eval_metric: curve}} if eval_sets else {}
        if self.store_history and not self.theta_history_:          
            self.theta_history_.append(theta.copy())
        self.best_score_ = best_score
        self.theta_ = best_theta if (self.early_stopping_used_ and best_theta is not None) else theta
        return self

    
    def fit_gradient_descent(self, X, y, sample_weight=None, verbose=False):
        prev = self.solver
        self.solver = "gd"
        try:
            return self._run(np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel(),
                             np.ones(len(y)) if sample_weight is None
                             else np.asarray(sample_weight, dtype=float), verbose=verbose)
        finally:
            self.solver = prev

    def fit_newton(self, X, y, sample_weight=None, verbose=False):
        prev = self.solver
        self.solver = "newton"
        try:
            return self._run(np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel(),
                             np.ones(len(y)) if sample_weight is None
                             else np.asarray(sample_weight, dtype=float), verbose=verbose)
        finally:
            self.solver = prev

    # -- project interface ----------------------------------------------------
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=None,
            eval_metric="logloss", verbose=False):

        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.feature_names_ = (list(feature_names) if feature_names is not None
                               else [f"x{j}" for j in range(X.shape[1])])
        w = self._sample_weights(y)

        if self.standardize:
            self.scaler_ = StandardScalerScratch().fit(X)
            Xs = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            Xs = X

        eval_sets = None
        if eval_set is not None:
            eval_sets = [(self.scaler_.transform(np.asarray(Xe, dtype=float)) if self.scaler_ is not None
                          else np.asarray(Xe, dtype=float), np.asarray(ye, dtype=int))
                         for Xe, ye in eval_set]

        self._run(Xs, y, w, eval_sets=eval_sets, early_stopping_rounds=early_stopping_rounds,
                  eval_metric=eval_metric, verbose=verbose)

        self.coef_ = self.theta_[1:].copy()
        self.intercept_ = float(self.theta_[0])
        self.fit_time_ = time.time() - t0
        imp = np.abs(self.coef_)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    # -- prediction -----------------------------------------------------------
    def decision_function(self, X):
        
        X = np.asarray(X, dtype=float)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.intercept_ + X @ self.coef_

    def predict_proba(self, X):
        
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        Xb = self._add_bias(X)
        for theta in self.theta_history_:
            yield Xb @ theta

    def staged_predict_proba(self, X):
        for F in self.staged_decision_function(X):
            p1 = sigmoid(F)
            yield np.column_stack([1.0 - p1, p1])

    def coefficient_table(self, top=None):
        rows = [{"feature": self.feature_names_[j], "coef_per_1SD": float(self.coef_[j]),
                 "odds_ratio": float(np.exp(self.coef_[j])), "importance": float(self.feature_importances_[j])}
                for j in range(len(self.coef_))]
        rows.sort(key=lambda r: -r["importance"])
        return rows[:top] if top is not None else rows

    def feature_importance_table(self, top=None, kind="gain"):
        return [(r["feature"], r["importance"]) for r in self.coefficient_table(top=top)]

    def export_text(self, top=5):
        lines = [f"z = {self.intercept_:+.4f}  (intercept, log-odds)", ""]
        for r in self.coefficient_table(top=top):
            lines.append(f"z += {r['coef_per_1SD']:+.4f} * z_{r['feature']:<22} "
                         f"(odds ratio {r['odds_ratio']:.3f} per +1 SD)")
        lines.append("")
        lines.append("P(y=1) = sigmoid(z)      [z_<feature> = the standardised feature value]")
        return "\n".join(lines)

    def pick_threshold(self, X_val, y_val, metric="f1"):
        proba = self.predict_proba(X_val)[:, 1]
        best_thr, best_score = best_threshold(y_val, proba, metric=metric)
        self.best_threshold_ = best_thr
        self.best_threshold_score_ = best_score
        return best_thr
