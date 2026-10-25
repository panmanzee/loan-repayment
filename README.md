# Loan Repayment: Machine Learning Project

ทำนาย 2 อย่างจากข้อมูลสินเชื่อ LendingClub (9,578 สัญญา):

- **งาน A: Regression**: ทายอัตราดอกเบี้ย (`int.rate`)
- **งาน B: Classification**: ทายว่าผู้กู้จะผิดนัดชำระหรือไม่ (`not.fully.paid`)

โมเดลทุกตัวเขียนเองด้วย NumPy และเทียบผลกับ scikit-learn
ผู้จัดทำ: panmanzee, pmipheta

## ผลลัพธ์หลัก

| งาน | โมเดลที่ดีที่สุด | คะแนน (ชุดทดสอบ) |
|---|---|---|
| A Regression | Gradient Boosting Regressor | R² = 0.760 |
| B Classification | Blend EBM + MLP | ROC-AUC = 0.7095 |

## โฟลเดอร์

```
DataSet/      ข้อมูล loan_data.csv
Notebooks/    โค้ดที่รันได้ + ผลลัพธ์ (1 notebook ต่อ 1 โมเดล)
  Regression/       00 วิเคราะห์ข้อมูล, 01-04 โมเดล, 05 เปรียบเทียบ
  Classification/   00 วิเคราะห์ข้อมูล, 01-15 โมเดล, 16 PCA/LDA, 17 Clustering, 18 เปรียบเทียบ
  src/              โค้ดโมเดลที่เขียนเอง (notebook import จากที่นี่)
Results/      กราฟ (.png) และตาราง (.csv) ของแต่ละโมเดล แยกตาม notebook
```

## เปิดดูผล

เปิดไฟล์ `.ipynb` ใน `Notebooks/` บน GitHub ได้เลย ผลรันและกราฟอยู่ในไฟล์แล้ว
กราฟและตารางของแต่ละโมเดลอยู่ที่ `Results/<งาน>/<ชื่อ notebook>/`

## รันเอง

```
pip install -r requirements.txt
python Notebooks/run_all_notebooks.py Regression       # งาน A (~15 นาที)
python Notebooks/run_all_notebooks.py Classification   # งาน B (~1.5 ชั่วโมง)
python Notebooks/run_all_notebooks.py 15_Blend         # notebook เดียว
```

แต่ละ notebook มี 3 ช่อง: **1. Setup** (โหลดและแบ่งข้อมูล) → **2. Model** (เทรนโมเดลที่เขียนเอง + scikit-learn) → **3. Results** (ตัวชี้วัด, กราฟ, บันทึกผล)
