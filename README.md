# Loan Repayment: ML Project

ข้อมูล `DataSet/loan_data.csv` (9,578 สัญญากู้ยืม, 14 คอลัมน์) โมเดลทุกตัวเขียนเองด้วย NumPy แล้วเทียบกับ scikit-learn
ผู้จัดทำ: panmanzee, pmipheta

โปรเจกต์มี 2 งานแยกกัน: **งาน A Regression** (ทำนายอัตราดอกเบี้ย) และ **งาน B Classification** (ทำนายการผิดนัดชำระ)
ทุกงานแบ่งข้อมูลเป็น train / validation / test, เลือก threshold และ hyper-parameter จาก validation เท่านั้น (ไม่แตะ test), และวัดซ้ำหลายรอบเป็น mean ± SD

---

# งาน A: Regression (เป้าหมาย `int.rate`)

**โมเดล:** Simple Linear (FICO), Multiple Linear, Polynomial (degree 2), **Gradient Boosting = better model** (โค้ดอยู่ใน `RegressionAlgorithms/`)

### ผลลัพธ์ตามหัวข้อที่ครูกำหนด

| โมเดล | Loss train (MSE) | Loss test (MSE) | RMSE | MAE | R² (test) | R² (5-fold CV) | R² (10 splits) |
|---|---|---|---|---|---|---|---|
| Baseline (เดาค่าเฉลี่ย) | 0.000728 | 0.000689 | 0.0263 | 0.0210 | -0.001 | - | -0.001 ± 0.001 |
| Simple Linear | 0.000357 | 0.000333 | 0.0183 | 0.0139 | 0.515 | 0.511 ± 0.015 | 0.512 ± 0.019 |
| Multiple Linear | 0.000240 | 0.000230 | 0.0152 | 0.0114 | 0.665 | 0.666 ± 0.018 | 0.671 ± 0.011 |
| Polynomial | 0.000308 | 0.000281 | 0.0168 | 0.0127 | 0.591 | 0.577 ± 0.017 | 0.579 ± 0.022 |
| **Gradient Boosting** | 0.000091 | 0.000165 | **0.0129** | **0.0090** | **0.760** | **0.770 ± 0.011** | **0.776 ± 0.009** |

**สรุป:** Gradient Boosting ดีที่สุดทุกตัวชี้วัด และชนะ Multiple Linear ครบทั้ง 10 splits (+0.105 R²) เป็นผลที่แน่นอน ไม่ใช่ความบังเอิญ

| หัวข้อ | ไฟล์ผลลัพธ์ |
|---|---|
| Loss / Loss curve | `Results/02_Loss/regression_*` |
| Performance curve | `Results/07_Performance_Curve/regression_*` |
| R², benchmark, batch runs | `Results/08_R_Square/regression_*` |
| Feature importance, residuals, เทียบ scikit-learn | `Results/09_Others/regression_*` |

---

# งาน B: Classification (เป้าหมาย `not.fully.paid`, ผิดนัด 16%)

**โมเดล:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Naive Bayes, Perceptron, Single-Layer Perceptron, MLP, EBM และ **Blend EBM + MLP = better model** (โค้ดอยู่ใน `ClassificationAlgorithms/`)

**Better model คืออะไร:** EBM (โมเดล additive แต่ละ feature มีกราฟของตัวเอง อธิบายเหตุผลรายคนได้) รวมกับ MLP 5 ตัวที่ seed ต่างกัน (bagging ลดความผันผวน) เฉลี่ยแบบ 50/50 ไม่มีการเทรนน้ำหนักจึงไม่ overfit

### ผลลัพธ์ตามหัวข้อที่ครูกำหนด (test set, threshold เลือกจาก validation)

| โมเดล | Loss (log-loss) | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.459 | 0.751 | 0.306 | 0.449 | 0.808 | 0.364 | 0.707 |
| Decision Tree | 0.520 | 0.590 | 0.221 | 0.623 | 0.583 | 0.326 | 0.632 |
| Random Forest | 0.447 | 0.620 | 0.244 | 0.662 | 0.612 | 0.357 | 0.688 |
| Gradient Boosting | 0.459 | 0.582 | 0.237 | 0.731 | 0.554 | 0.358 | 0.693 |
| XGBoost | 0.456 | 0.545 | 0.222 | 0.744 | 0.507 | 0.342 | 0.684 |
| Naive Bayes | 0.657 | 0.683 | 0.250 | 0.498 | 0.718 | 0.333 | 0.676 |
| Perceptron (averaged) | 0.463 | 0.525 | 0.225 | 0.810 | 0.471 | 0.352 | 0.691 |
| Single-Layer Perceptron | 0.460 | 0.694 | 0.281 | 0.593 | 0.713 | 0.382 | 0.705 |
| MLP | 0.461 | 0.712 | 0.290 | 0.557 | 0.742 | 0.382 | 0.709 |
| EBM | 0.459 | 0.527 | 0.225 | 0.810 | 0.473 | 0.353 | 0.705 |
| **Blend EBM + MLP** | 0.458 | 0.700 | 0.289 | 0.607 | 0.718 | **0.392** | 0.708 |

Baseline "ทุกคนจ่ายคืน" ได้ Accuracy 0.84 แต่ Sensitivity 0 (จับคนผิดนัดไม่ได้เลย) จึงดู Accuracy อย่างเดียวไม่ได้

### Batch runs (สุ่ม split 10 ครั้งที่ไม่เคยใช้เลือกโมเดล, mean ± SD)

| โมเดล | ROC-AUC | PR-AUC |
|---|---|---|
| Random Forest | 0.665 ± 0.013 | 0.273 ± 0.011 |
| Gradient Boosting | 0.669 ± 0.012 | 0.282 ± 0.012 |
| Logistic Regression | 0.674 ± 0.013 | 0.290 ± 0.013 |
| EBM | 0.674 ± 0.013 | 0.290 ± 0.013 |
| MLP | 0.676 ± 0.015 | 0.287 ± 0.013 |
| **Blend EBM + MLP** | **0.677 ± 0.014** | **0.290 ± 0.014** |

**สรุป:** Blend ดีกว่า Logistic Regression +0.003 (ชนะ 9 จาก 10 ครั้ง) และดีกว่า Random Forest +0.011 ต่างกันเล็กน้อยเพราะข้อมูลชุดนี้มี noise สูง ทุกโมเดลได้ AUC ราว 0.67-0.69 (ลองปรับ regularization, เพิ่ม feature, เปลี่ยนโมเดล ก็ไม่เกินนี้) จุดเด่นของ Blend คือเสถียรกว่าและอธิบายเหตุผลรายคนได้

| หัวข้อ | ไฟล์ผลลัพธ์ |
|---|---|
| Loss curve | `Results/02_Loss/classification_*` |
| Confusion matrix | `Results/03_Confusion_Matrix/` |
| Accuracy | `Results/04_Accuracy/` |
| Precision, Sensitivity, Specificity, F1 | `Results/05_Precision_Recall_F1/` |
| ROC & AUC | `Results/06_ROC_AUC/` |
| Performance curve | `Results/07_Performance_Curve/classification_*` |
| Batch runs, feature importance, EBM shape functions, เทียบ scikit-learn | `Results/09_Others/classification_*` |

---

## โครงสร้างโฟลเดอร์

| ชื่อ | คืออะไร |
|---|---|
| `Common/` | โหลด/แบ่งข้อมูล, ตัววัดผล, ฟังก์ชันวาดกราฟ |
| `RegressionAlgorithms/` | โมเดลงาน A |
| `ClassificationAlgorithms/` | โมเดลงาน B |
| `run_regression.py` / `run_classification.py` | รันงาน A / งาน B ทั้งหมด (ไฟล์ละประมาณ 15 นาที) |
| `Results/` | กราฟและตาราง แยกตามหัวข้อ `01`-`09` |
| `TrainedModels/` | โมเดลที่เทรนแล้ว (`.joblib`) |

## วิธีรัน

```bash
pip install numpy pandas matplotlib scikit-learn joblib
python run_regression.py
python run_classification.py
```
