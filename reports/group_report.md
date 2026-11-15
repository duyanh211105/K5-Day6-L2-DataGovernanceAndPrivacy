# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Mọi con số dưới đây lấy từ file do script sinh ra (`out/*/shift_scores.csv`, `out/*/run_info.json`,
> `reports/t10_label_free_summary.md` do `python -m tools.t10_summary` tạo). Repo học viên **không có GT**, nên
> metric chính thức (Spearman ρ, AUROC, false-alarm, recall) để trống, chờ giảng viên chạy `make eval-baseline eval-poc compare TRACK=t10`.

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G00 (làm một mình; cập nhật nếu lớp cấp mã) |
| Challenge | T10 · Domain Shift Radar & Slice Dashboard (chọn ngẫu nhiên `random.randint(1,14)` theo yêu cầu lab coach) |
| Thành viên + vai trò | 1 người: duyanh211105, đảm nhận cả Lead, Core algorithm, QA/metric, Review và report |
| Git URL | ____ |
| Tag `demo-180m` (commit) | ____ |
| Tag `final-24h` (commit) | ____ |
| Deadline (giờ VN) | ____ |

## 01 · Problem statement

- **Ai chịu hậu quả khi sai:** người đi bộ và đội vận hành. Nếu một batch dữ liệu mới (đêm, mưa, camera khác…) làm model
  phát hiện người đi bộ tụt hiệu năng mà không ai biết, model bị ship vào điều kiện nó không xử lý được. Ngược lại,
  báo động nhầm thì tốn chi phí gán nhãn lại và retrain vô ích.
- **Điểm nghẽn:** cảnh báo drift thông thường đo "ảnh trông khác", trong khi thứ cần đo là "model có tệ đi không",
  và lúc deploy thì **không có nhãn**.
- **Đầu vào → đầu ra:** 6 batch × 40 ảnh + `pred_score` của model → `shift_scores.csv`
  (`batch_id, shift_score, pred_high_risk`), cùng review queue, dashboard và quyết định GO/HOLD.
- **Tiêu chí (chốt trước eval, `configs/tracks/t10.yaml`):** HOLD batch khi z(model drift) ≥ 1.645; REVIEW khi
  z(image drift) ≥ 3 mà không có bằng chứng phía model; KEEP cho trường hợp còn lại. Release = GO chỉ khi không batch nào HOLD/REVIEW.
- **Giả định cần domain owner xác nhận:** `pred_score` là output thật của model đang deploy; `b0_source` đại diện đúng
  phân bố train; tụt hiệu năng sẽ phần nào lộ ra trong phân bố `pred_score`.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| reference (`b0_source`) | 40 ảnh | `fixtures/t10_domain_shift` (tools/make_fixtures.py, seed 42) | CC0-1.0 | có |
| eval (5 batch mới) | 200 ảnh | như trên | CC0-1.0 | có |

- Schema: `images.csv (image_id, batch_id, path)` + `model_predictions.csv (image_id, batch_id, pred_score, pred_label)`, ảnh 256×144.
- Seed 42. Split theo batch (điều kiện chụp), không trộn ảnh giữa các batch.
- `manifest.csv` verified (sha256): **có**. `load()` gọi `verify_manifest`, lệch hash thì exit 2 (có test).
  Lưu ý trên Windows: `core.autocrlf=true` đổi CSV sang CRLF làm lệch sha256. Đã tắt `core.autocrlf` cho repo.
- Không dùng dữ liệu thật.

## 03 · Research & design

- **MMD với kernel RBF + permutation test** (Gretton et al., *A Kernel Two-Sample Test*, JMLR 2012): kiểm định hai mẫu
  đa chiều, không giả định phân bố. Dùng cho 8 descriptor điều kiện ảnh.
- **KS test trên output model / "black-box shift detection"** (Rabanser et al., *Failing Loudly*, NeurIPS 2019;
  Lipton et al., *BBSE*, ICML 2018): drift trên output model gần với hiệu năng hơn drift pixel.
- **Evidently DataDriftPreset:** tham khảo cách báo drift theo cột (per-feature). Không cài vào pipeline để giữ offline và ít dependency.
- **Data flow:**
  `manifest check → 8 descriptor/ảnh + uncertainty → [MMD² ảnh | KS pred_score] vs b0_source, permutation z →
  shift_score = 0.6·s_pred + 0.4·s_img → slice scan → queue (6 batch + 14 ảnh) → reviewer override + audit →
  KEEP/REVIEW/HOLD từng batch + release decision → dashboard.html`
- Không dùng model hay weights pretrained. Chỉ OpenCV, SciPy và NumPy.
- **Trade-offs:**
  - Image drift luôn "cực kỳ có ý nghĩa" (z 32–51) vì các batch khác nhau rõ về thị giác, nên nó không tự phân biệt được batch nguy hiểm.
  - Model drift sát mục tiêu hơn nhưng N = 40 quá ít để KS phát hiện thay đổi nhỏ.
  - Vì vậy: chỉ model drift mới được kích hoạt HOLD; image drift chỉ đưa batch vào REVIEW.

## 04 · Baseline vs PoC

- **Baseline** (`src/baseline_t10.py`): |mean brightness(batch) − mean(source)| / sd(source); flag khi ≥ 2 sd.
  Dùng cùng `load()`, cùng pool và seed.
- **PoC** (`src/pipeline.py` + `src/t10_features.py`): như mục 03, tham số trong `configs/tracks/t10.yaml`.
- Cùng 6 batch, cùng reference, cùng seed 42: **xác nhận**.
- Ablation trong cùng run: `out/poc/ablation_image_only.csv` (chỉ s_img) và `out/poc/ablation_model_only.csv` (chỉ s_pred).
  Giảng viên chấm được cả hai bằng `python -m src.evaluate --track t10 --pred <file>`.

**Bảng 1 · Baseline vs PoC.** Metric chính thức chờ GT. Bảng dưới là số label-free từ `reports/t10_label_free_summary.md`:

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Spearman ρ(ShiftScore, PerfDrop) | chờ GT | chờ GT | `make eval-baseline eval-poc compare TRACK=t10` |
| Thứ hạng batch (1 = shift lớn nhất) | night > dusk > rain > camera > city | rain > camera > dusk > night > city | ρ giữa hai ranking = 0.486 (N = 6) |
| Số batch flag high-risk | 3 (night, rain, dusk) | 0 | PoC không thấy model drift có ý nghĩa ở batch nào (min p = 0.289) |
| Runtime / run (6 batch, 240 ảnh) | 0.49 s | 2.74 s | CPU AMD64 Family 25, 300 permutation × 2 thống kê × 5 batch |

## 05 · Evaluation protocol

- Công thức, đơn vị và denominator: xem `reports/metric_contract.md`.
- Tất cả ngưỡng đặt a priori theo thống kê (1.645 = một phía 5%; 3 sd). Không tune, vì không có nhãn.
- Độ bất định: permutation p-value cho từng batch (300 lần) và bootstrap CI cho ρ (evaluator). Với N = 6, CI của ρ
  sẽ rất rộng; nếu CI chứa 0 thì không kết luận được gì.
- Lệnh evaluate:
  `python -m src.evaluate --track t10 --pred out/poc/shift_scores.csv --run-name poc --out reports/metrics_poc.json`

## 06 · Result slices

| Slice | N | Baseline score | PoC score | Ghi chú |
|---|---|---|---|---|
| b1_night | 40 | 23.00 (flag) | 0.081 (REVIEW) | lệch sáng lớn nhất, model không đổi (KS p = 0.92) |
| b2_rain | 40 | 4.49 (flag) | 0.208 (REVIEW) | KS cao nhất (0.200) nhưng p = 0.29 |
| b3_camera | 40 | 1.81 | 0.175 (REVIEW) | độ nét −41 sd (blur), baseline gần như không thấy |
| b4_city | 40 | 0.78 | 0.079 (REVIEW) | lệch ảnh nhỏ nhất, chủ yếu ở saturation |
| b5_dusk_far_peds | 40 | 7.65 (flag) | 0.153 (REVIEW) | pos_rate 0.40 so với 0.50 ở source; uncertainty cao nhất (0.489) |
| brightness / sharpness tertile × batch | 25 ô | – | 0 ô silent_risk | ô có uncertainty gap lớn nhất: dusk·sharp, n = 2 (quá nhỏ để flag) |

- **Failure case 1: shift lớn nhưng model không đổi (`b1_night`, `demo/t10_failure1_b1_night.jpg`).**
  Độ sáng −23 sd, `dark_frac` +983 sd. Baseline xếp hạng 1 và flag high-risk. Nhưng phân bố `pred_score` gần như
  trùng source (KS = 0.125, p = 0.92) và uncertainty còn thấp hơn source (0.447 so với 0.467).
  *Root cause:* người đi bộ có màu sáng trên nền tối nên vẫn tương phản cao; độ sáng toàn cục không phải biến
  làm model sai. PoC đưa batch này vào REVIEW (nghi false alarm) chứ không HOLD.
- **Failure case 2: rủi ro âm thầm (`b5_dusk_far_peds`, `demo/t10_failure2_b5_dusk_far_peds.jpg`).**
  Người đi bộ ở xa, chỉ vài pixel. Model nói "có người" ít hơn (pos_rate 0.40 so với 0.50) và kém chắc chắn nhất (0.489).
  Dù vậy KS chỉ đạt p = 0.49 và slice scan không flag gì.
  *Root cause:* shift nằm ở **kích thước đối tượng**, không ở điều kiện ảnh. Descriptor toàn ảnh và tertile
  sáng/nét không tách ra được slice "người xa"; với N = 40, giảm 4/40 dự đoán dương không đủ ý nghĩa thống kê.
  Đây là điểm mù chính của PoC.
- **Latency:** PoC 2.74 s / 240 ảnh ≈ 11 ms/ảnh (gồm permutation test); baseline 0.49 s.

## 07 · Human & decision

- **Review queue** `out/poc/review_queue.csv`, B = 20: 6 dòng batch (evidence = montage 8 ảnh lạ nhất) + 14 ảnh có
  atypicality × (0.5 + uncertainty) cao nhất. Có giao diện offline `out/poc/contact_sheet.html` với nút Export CSV.
- **Override** (`review()` trong `src/pipeline.py`):
  - `batch:*`: `edit` với giá trị 0/1, `reject` → 0, `escalate` → 1.
  - `img:*`: `escalate` → batch của ảnh đó thành HOLD.
  - Mọi dòng review được ghi vào `out/poc/audit_log.jsonl` (actor, before/after, note). Giá trị override sai thì exit 2 (có test).
- **Decision:** từng batch được ghi ra `out/poc/decisions/<batch>.json` và `decision_log.csv` (KEEP / ESCALATE / HOLD,
  kèm reason_code và owner). Run hiện tại cho release = **HOLD · `PENDING_HUMAN_REVIEW`**, vì 5 batch có shift ảnh lớn
  nhưng không có bằng chứng model drift.
- **Khi nào HOLD thay vì tự release:** khi có model drift có ý nghĩa, hoặc reviewer escalate. Image drift lớn thì phải có
  người ký trước khi release, vì PoC không chứng minh được batch đó an toàn.

## 08 · Reproduce & tests

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt      # Windows; Linux: .venv/bin/python
git config core.autocrlf false                                                        # Windows: giữ sha256 của fixtures
python -m src.main --input fixtures/t10_domain_shift --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42
python -m src.main --track t10 --pipeline src.baseline_t10 --output out/baseline --run-name baseline --offline --seed 42
python -m tools.t10_summary                                                           # reports/t10_label_free_summary.md
python -m src.evaluate --track t10 --pred out/poc/shift_scores.csv --out reports/metrics_poc.json   # cần GT giảng viên
pytest -q
```

- `pytest`: **56 passed, 39 skipped** (các test cần GT), 0 failed. Test của T10 nằm ở `tests/test_t10_pipeline.py`
  (9 test: output contract, tái lập cùng seed, baseline, ảnh bị sửa → exit 2, thiếu prediction → exit 2, null calibration,
  phát hiện shift thật, override + audit, override sai → exit 2).
- Máy: Windows 11, AMD64 Family 25 (AMD), Python 3.12.10.

## 09 · Limitations & ethics

- Dữ liệu synthetic CC0, không có người hay biển số thật. Không có rủi ro PII trong repo. Nếu áp dụng cho dữ liệu thật
  (BDD100K…), ảnh không được commit hay đưa lên cloud.
- **Điều chưa biết:** chưa có GT nên chưa biết ρ thật. Model-drift signal hiện **không có ý nghĩa ở batch nào**, nên
  ranking PoC thực tế do image drift quyết định. Rất có thể ρ của PoC không tốt hơn baseline; điều này sẽ được báo trung thực khi có số.
- False alarm: PoC đưa cả 5 batch vào REVIEW. Ở quy mô 10× (60 batch), reviewer sẽ quá tải, nên cần một bước
  ưu tiên REVIEW tốt hơn.
- Không tuyên bố tuân thủ GDPR hay luật Việt Nam. Đây là công cụ kỹ thuật; quyết định release cần owner và domain expert ký.
- **Kế hoạch cải tiến:**
  1. Descriptor theo đối tượng (kích thước blob người, contrast cục bộ) để bắt "người xa".
  2. Embedding CNN thay descriptor thủ công (ablation histogram vs embedding).
  3. Gom nhiều batch nhỏ (streaming) để tăng N cho KS.
  4. Gán nhãn tay 10 ảnh/batch để có proxy accuracy.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| duyanh211105 (một mình) | duyanh211105 | Toàn bộ: pipeline, features, baseline, tests, report | xem `git log --author=duyanh211105` | có |

## 11 · References

| URL | Ngày xem | Ý nghĩa |
|---|---|---|
| https://jmlr.org/papers/v13/gretton12a.html | (cần mở xác nhận) | MMD và kiểm định hai mẫu bằng kernel: cơ sở cho image drift |
| https://arxiv.org/abs/1810.11953 | (cần mở xác nhận) | *Failing Loudly*: so sánh các cách phát hiện shift; output model + KS là baseline mạnh |
| https://arxiv.org/abs/1802.03916 | (cần mở xác nhận) | BBSE: dùng output model black-box để phát hiện label shift |
| https://docs.evidentlyai.com | (cần mở xác nhận) | DataDriftPreset: cách trình bày drift theo cột trên dashboard |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics (chờ GT), [x] ≥1 failure case, [x] reviewer evidence, [x] decision log.
- [x] Có license/source provenance, data splits, seed, offline environment, N.
- [x] Thành viên có commit thực chất và được nêu minh bạch trong bảng trên.
