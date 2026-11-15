# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T10 · Domain Shift Radar |
| Primary metric (tên đúng như evaluator in ra) | `Spearman ρ(ShiftScore, PerformanceDrop)` kèm 95% bootstrap CI (2000 lần) |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | batch (N = 6, gồm cả `b0_source`) |
| Công thức + numerator / denominator | Spearman rank correlation giữa `shift_score` của nhóm và `perf_drop` (GT ẩn) trên 6 batch. Phụ: AUROC high-risk (drop ≥ 0.10); False-alarm rate = batch không rủi ro bị flag / số batch không rủi ro; High-risk recall = batch rủi ro bị flag / số batch rủi ro |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | ghép theo `batch_id` (khớp chính xác, mỗi batch đúng 1 dòng) |
| Candidate pool cố định | 6 batch × 40 ảnh trong `fixtures/t10_domain_shift` (manifest sha256 đã verify) |
| Review budget B / K | B = 20 dòng review: cả 6 batch + 14 ảnh lạ nhất (atypicality × uncertainty) |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Không có nhãn → không tune. Ngưỡng chọn theo thống kê, a priori: `pred_high_risk` = z(model drift) ≥ 1.645 (permutation test một phía 5%); REVIEW khi z(image drift) ≥ 3; trọng số w_pred = 0.6, w_img = 0.4 (ưu tiên tín hiệu gần hiệu năng model). Baseline: flag khi lệch độ sáng ≥ 2 sd. Tất cả nằm trong `configs/tracks/t10.yaml` |
| Eval split (không dùng để tune) | Toàn bộ 6 batch là eval; GT `gt/batch_perf.csv` chỉ giảng viên chạy |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Xấu nhất: **miss** một batch tụt hiệu năng (ship model hỏng). Thứ hai: false alarm (HOLD vô ích, tốn chi phí relabel) |
| Slices sẽ báo cáo (≥2) | (1) theo batch/điều kiện; (2) tertile độ sáng và tertile độ nét trong từng batch (biên tertile lấy từ source) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | bootstrap CI cho ρ (evaluator in); permutation p-value cho từng batch. CI chứa 0 → **không** kết luận score dự báo được drop |
| Ngày giờ chốt + người chốt | 2026-11-15, duyanh211105 (làm một mình) |

Thay đổi thiết kế trước khi chốt (không dùng GT): v1 chuẩn hoá thành phần bằng "chia cho max các batch", khiến nhiễu
(mọi z model < 1) bị phóng thành full scale. v2 (bản chốt) dùng thang cố định: s_pred = z/3, s_img = MMD²/2.

Không có GT trong repo học viên nên baseline/PoC chưa có số chính thức; nếu baseline = 0 thì ghi rõ "baseline = 0".
