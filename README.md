## ไฟล์และโฟลเดอร์

| ชื่อ | คืออะไร |
|---|---|
| `DataSet/` | ไฟล์ข้อมูล `loan_data.csv` |
| `Common/data.py` | จัดการข้อมูล: โหลด, สร้าง feature, แบ่ง train/test, scale, k-fold |
| `Common/metrics.py` | ตัววัดผล: loss (MSE), RMSE, MAE, R², feature importance |
| `RegressionAlgorithms/` | โค้ดโมเดล regression 4 ตัว (ที่ยังไม่เทรนกับข้อมูลของเรา): Linear, Multiple, Polynomial, Gradient Boosting (Better Model) |
| `run_regression.py` | **ไฟล์รัน** เอาข้อมูล + โมเดล + ตัววัดผลมารวมกัน แล้วพิมพ์ผลของทุกโมเดล (เรียกใช้ data.py และ metrics.py) |
| `TrainedModels/` | โมเดลที่เทรนเสร็จแล้ว เก็บเป็นไฟล์ (`regression_model.joblib`) |
| `Results/` | ผลลัพธ์ทุกตารางเป็นไฟล์ CSV (01 วิเคราะห์ข้อมูล, 02 loss/R²/RMSE/MAE ทุกโมเดล, 03 เทียบกับ sklearn, 04 cross-validation, 05 loss curve, 06 performance curve, 07 feature importance) เอาไปทำกราฟ/สไลด์ได้ |

โฟลเดอร์ `...Algorithms` คือ "สูตร" ยังไม่เคยเรียนรู้จากข้อมูล ส่วน `TrainedModels` คือโมเดลที่เทรนแล้ว

## วิธีรัน

```bash
pip install numpy pandas scikit-learn joblib
python run_regression.py
```

ใช้เวลาประมาณ 1–2 นาที จะพิมพ์ผลเป็นหัวข้อ 1–10 (วิเคราะห์ข้อมูล → เทรน → loss / R² → cross-validation → feature importance → บันทึก/โหลดโมเดล)
`sklearn` ใช้เพื่อเทียบผลกับโมเดลที่เขียนเองเท่านั้น
