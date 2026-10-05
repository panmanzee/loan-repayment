# loan-repayment

โปรเจกต์ ML วิชา 01076641 ใช้ข้อมูลสินเชื่อ LendingClub (`DataSet/loan_data.csv`)
- **Regression:** ทำนายอัตราดอกเบี้ย `int.rate`
- **Classification:** ทำนายว่าผิดนัดชำระไหม `not.fully.paid`

## ไฟล์และโฟลเดอร์

| ชื่อ | คืออะไร |
|---|---|
| `DataSet/` | ไฟล์ข้อมูล `loan_data.csv` |
| `Common/data.py` | จัดการข้อมูล: โหลด, สร้าง feature, แบ่ง train/test, scale, k-fold |
| `Common/metrics.py` | ตัววัดผล: loss (MSE), RMSE, MAE, R², feature importance |
| `RegressionAlgorithms/` | โค้ดโมเดล regression 4 ตัว (เขียนเองทั้งหมด): Linear, Multiple, Polynomial, Gradient Boosting (โมเดล better) |
| `ClassificationAlgorithms/` | โค้ดโมเดล classification 12 ไฟล์ ตามลิสต์ของครู (ตอนนี้ 01 กับ 03 เป็น sklearn, ที่เหลือยังเป็นโครงเปล่า) |
| `run_regression.py` | **ไฟล์รัน** เอาข้อมูล + โมเดล + ตัววัดผลมารวมกัน แล้วพิมพ์ผลของทุกโมเดล |
| `TrainedModels/` | โมเดลที่เทรนเสร็จแล้ว เก็บเป็นไฟล์ (`regression_model.joblib`) โหลดมาทำนายได้เลย |
| `Results/` | ตารางผลลัพธ์ (`regression_results.csv`) |

โฟลเดอร์ `...Algorithms` คือ "สูตร" ยังไม่เคยเรียนรู้จากข้อมูล ส่วน `TrainedModels` คือโมเดลที่เทรนแล้ว

## วิธีรัน

```bash
pip install numpy pandas scikit-learn joblib
python run_regression.py
```

ใช้เวลาประมาณ 1–2 นาที จะพิมพ์ผลเป็นหัวข้อ 1–10 (วิเคราะห์ข้อมูล → เทรน → loss / R² → cross-validation → feature importance → บันทึก/โหลดโมเดล)
`sklearn` ใช้เพื่อเทียบผลกับโมเดลที่เขียนเองเท่านั้น

## สำหรับเพื่อนที่ทำ Classification

1. เขียนโมเดลในโฟลเดอร์ `ClassificationAlgorithms/` (ไฟล์ละ 1 โมเดล ตามเลขนำหน้า)
2. สร้าง `run_classification.py` ตามรูปแบบเดียวกับ `run_regression.py`
3. ฟังก์ชันเตรียม feature ของ classification (ต้องใช้ `int.rate`, `fico_rate_gap`, แบ่งแบบ `stratify`) ให้เขียนเพิ่มใน `Common/data.py` เป็นฟังก์ชันใหม่ อย่าแก้ฟังก์ชัน regression
4. ตัววัดผลของ classification (confusion matrix, specificity, F1, ROC-AUC) ให้สร้างไฟล์ใหม่ `Common/classification_metrics.py` จะได้ไม่ชนกับ `metrics.py`
