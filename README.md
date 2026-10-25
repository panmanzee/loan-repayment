# Loan Repayment: Machine Learning Project

ทำนาย 2 อย่างจากข้อมูลสินเชื่อ LendingClub (9,578 สัญญา)

- งาน A Regression: ทายอัตราดอกเบี้ย (`int.rate`)
- งาน B Classification: ทายว่าผู้กู้จะจ่ายคืนไม่ครบหรือไม่ (`not.fully.paid`)

ทุกโมเดลเขียนเองด้วย NumPy และเทียบผลกับ scikit-learn

## ผลลัพธ์

| งาน | โมเดลที่ดีที่สุด | คะแนน (test) |
|---|---|---|
| A Regression | Gradient Boosting Regressor | R² = 0.760 |
| B Classification | Blend EBM + MLP | ROC-AUC = 0.7095 |

## โฟลเดอร์

| โฟลเดอร์ | มีอะไร |
|---|---|
| `DataSet/` | ข้อมูล `loan_data.csv` |
| `Model/` | โค้ดโมเดลทั้งหมด (.py) แยกเป็น `Regression/`, `Classification/` และ `Common/` (โค้ดที่ใช้ร่วมกัน) |
| `Notebooks/` | notebook 1 ไฟล์ต่อ 1 โมเดล มีผลรันอยู่ในไฟล์แล้ว |
| `Results/` | กราฟและตารางของแต่ละ notebook |

## รัน

```
pip install -r requirements.txt
python Model/run_all_notebooks.py Regression
python Model/run_all_notebooks.py Classification
```

ผู้จัดทำ: panmanzee, pmipheta
