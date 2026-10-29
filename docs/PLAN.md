# AVM Training — Implementation Plan (v0, dữ liệu HF)

Nguồn: `v-nexus-backend/specs/avm/002-model-training/plan.md` §9. Kiến trúc: [ARCHITECTURE.md](ARCHITECTURE.md).
Mỗi stage = 1 module + 1 lệnh CLI + test riêng. Làm tuần tự S0 → S9; mỗi stage xong thì chạy được độc lập trên output stage trước.

Quy ước chung: output ở `data/<run_id>/<stage>/` (parquet + `*_report.json`); stage ghi `_DONE` khi xong; chạy lại stage ghi đè; test dùng fixture nhỏ tổng hợp (không dùng dữ liệu thật).

---

## S0. Nền tảng (`cli.py`, `config.py`, `io.py`)

- **Làm**: argparse `avm-train <stage> --config`, loader YAML + validate (khóa bắt buộc, ngày hợp lệ), helper đường dẫn run dir, đọc/ghi parquet, logger (không log giá trị cá nhân/địa chỉ thô).
- **Test**: config thiếu khóa → lỗi rõ; stage lạ → exit khác 0.
- **Xong khi**: `avm-train fetch --config ...` dispatch được tới stage (dù stage chưa làm).

## S1. fetch (`extract/hf.py`)

- **Vào**: `sources.*` trong config. **Ra**: `raw/listings.parquet`, `raw/crosswalk.csv`, `fetch_meta.json` (revision, commit, sha256, số dòng).
- **Làm**: `huggingface_hub` tải đúng `hf_revision`; tải CSV crosswalk đúng `crosswalk_commit`; giữ 19 cột cần dùng; lọc `published_at` trong cửa sổ split.
- **Chặn**: `hf_revision` hoặc `crosswalk_commit` là null → raise, không tải.
- **Chưa kiểm**: nguồn tải thật của CSV crosswalk (HF hay GitHub) và tên cột thực tế. Kiểm khi làm, ghi vào `fetch_meta.json`.
- **Test**: pin null bị từ chối; (mock) meta ghi đúng.

## S2. crosswalk (`extract/crosswalk.py`)

- **Vào**: raw. **Ra**: `crosswalk/listings.parquet` (+ `ward_code_new`, `ward_code_db`, `province_code_new`, `ward_confidence`), `crosswalk_report.json`.
- **Làm**: chuẩn hóa tên 2 phía (bỏ dấu, hạ chữ, bỏ tiền tố, gộp khoảng trắng; quận ghi số → "Quận N"); khớp theo (tỉnh, quận, phường) cũ; zero-pad mã 5 chữ số; gắn `exact` / `default_split`; dòng không khớp bị bỏ. Xác định mã HCM mới từ CSV rồi ghi vào config `province_code_new`.
- **Báo cáo**: tỉ lệ khớp, tỉ lệ `default_split`, top 30 khóa không khớp theo số dòng, số dòng theo `province_code_new`.
- **Cổng dừng**: tỉ lệ khớp (mẫu số = dòng đủ tỉnh/quận/phường) < `min_match_rate` → exit 2. Phường mới HCM không ánh xạ 1–1 sang `wards.code` (khớp tên giữ dấu) → exit 2.
- **Dòng thiếu quận/phường (15,2%)**: loại bỏ ở v0, báo `rows_missing_location`. Lý do và các cách cứu đã đo: plan BE §9.13.
- **Đối chiếu DB**: `--wards-file` (CSV từ `scripts/export_wards.py`, đọc DB qua biến `AVM_DB_*`, chỉ SELECT). Bắt buộc.
- **Test**: bảng nhỏ tổng hợp: khớp exact, split mặc định, quận số, tên biến thể, không khớp.

## S3. clean (`clean/rules.py`)

- **Vào**: crosswalk. **Ra**: `clean/listings.parquet`, `clean_report.json` (số dòng bị loại theo từng luật, tỉ lệ giá < 100 triệu).
- **Thứ tự**: lọc `province_code_new` = HCM → bỏ price null/0 → giá trong [100 triệu, 500 tỷ] → area ngoài [5, 50.000] bỏ dòng → ánh xạ `ptype_group` (§9.5, loại 3 loại ngoài phạm vi) → khử trùng theo (`ward_code_db`, `street_name`, `area_m2`, `price`, ngày) → loại ngoại lai MAD(`log(price/area)`) > 4 theo (`ward_code_db`, `ptype_group`), ô thưa dùng `ptype_group` toàn HCM.
- **Quyết định tôi tự đặt (cần bạn xác nhận)**: plan ghi "kẹp" cho `frontage_m` (1–50) và `road_width_m` (0,5–60). Giá trị ngoài khoảng là lỗi nhập (max 9.080.000), kẹp về 50 sẽ thành số sai. Tôi đặt giá trị ngoài khoảng → NaN (cây xử lý NaN gốc), không kẹp.
- **Test**: mỗi luật 1 case; MAD với ô thưa; dedup giữ đúng dòng.

## S4. split (`split.py`)

- **Vào**: clean. **Ra**: `split/{train,val,test_a}.parquet`, cột `fold` (0–4) trên train.
- **Làm**: chia theo `published_at` theo config; `GroupKFold(5)` theo `ward_code_db` trên train. Test B không thuộc stage này (lấy từ DB ở S8).
- **Kiểm**: không phường nào nằm ở 2 fold; không dòng nào lọt 2 tập; ngày các tập không chồng.
- **Test**: hai kiểm trên bằng dữ liệu tổng hợp.

## S5. features (`features/normalize.py`, `price_ref.py`, `schema.py`)

- **Vào**: split. **Ra**: `features/{train,val,test_a}.parquet` (13 feature + `y=log(price)`), `feature_schema.json` (`fs-hf-0`).
- **normalize**: `house_direction` về 8 hướng + null (gộp "Đông Nam"/"Đông - Nam"…); `street_name` bỏ dấu; `has_project`. Một hàm duy nhất, dùng lại cho Test B và làm fixture cho Java.
- **price_ref**: `ward_price_ref` và `street_price_ref` (target encoding làm mượt) tính **trong từng fold** từ phần train của fold; ô < 20 dòng rơi về trung vị `ptype_group` rồi trung bình toàn cục. Bảng cuối tính trên toàn train, ghi vào `feature_schema.json`.
- **Cố ý**: `street_name` thô không vào X; chỉ qua `street_price_ref`.
- **Test**: price_ref không dùng dòng của fold đang đánh giá (rò rỉ); ô thưa fallback đúng; category lạ → NaN.

## S6. baseline (`train/baseline.py`)

- **Vào**: features. **Ra**: `baseline_metrics.json`.
- **Làm**: trung vị `price/area` theo (`ward_code_db`, `ptype_group`) × `area`; ô thưa fallback như S5.
- **Xong khi**: có MdAPE/PPE15 baseline trên val và test A để model phải vượt.

## S7. train + CQR (`train/lgbm.py`, `train/cqr.py`)

- **Vào**: features. **Ra**: `models/model_{point,q05,q95}.txt`, `conformal.json`.
- **Làm**: 3 model LightGBM (L2 trên log price, quantile 0,05 và 0,95) cùng feature, category khai báo rõ, seed cố định.
- **Phân vai tập (tôi đề xuất)**: fit trên train trừ 2 tuần cuối (early stopping), calibration CQR trên val (2026-02), test A chỉ để đánh giá cuối. Tránh dùng cùng tập cho early stopping và calibration.
- **CQR**: `q̂` theo phân đoạn (`ptype_group` × `ward_code_db`), phân đoạn có < ngưỡng dòng rơi về `ptype_group`; cận = `[q05 − q̂, q95 + q̂]`.
- **Test**: model nạp lại cho dự đoán giống hệt; coverage trên calibration ≈ 90%; thứ tự q05 ≤ point ≤ q95 sau CQR (cắt nếu vi phạm).

## S8. evaluate + gate (`evaluate/metrics.py`, `gate.py`)

- **Vào**: models, test A, folds, Test B. **Ra**: `metrics.json`, exit code gate.
- **Metric**: MdAPE, PPE10, PPE15, COD, PRD, coverage. Phân rã: toàn cục, `ptype_group`, `ward_confidence`, phường.
- **Đánh giá**: Test A; OOF group-kfold theo phường (huấn luyện lại từng fold); Test B (dự đoán `listings.price` SALE trong DB bằng cùng hàm normalize, đọc qua replica read-only, `extract/vnexus_db.py`).
- **Gate**: Test A MdAPE ≤ 15% và PPE15 ≥ 50% → pass; Test B > 25% chỉ cảnh báo. Nếu `default_split` kém hẳn: chạy lại không có nhóm này và so sánh.
- **Test**: công thức metric trên số tay tính; gate trả exit code đúng.
- **Chặn**: Test B cần quyền đọc replica. Chưa có thì stage vẫn chạy phần còn lại và ghi `test_b: skipped`.

## S9. export (`export/artifacts.py`)

- **Vào**: tất cả stage trước. **Ra**: `artifacts/<version>/`.
- **Gồm**: 3 model, `conformal.json`, `feature_schema.json`, `metrics.json`, `crosswalk_report.json`, `golden.csv` (~500 dòng: input, điểm, cận, SHAP bằng `pred_contrib`), `manifest.json` (revision HF, commit CSV, seed, git hash, sha256 từng tệp, `NON_PRODUCTION: true`).
- **Hoãn**: `oof_predictions.csv` thuộc T3 (production); v0 không xuất.
- **Từ chối**: gate fail → không export (trừ `--force-draft`, gắn `DRAFT`).
- **Test**: sha256 trong manifest khớp tệp; manifest có cờ `NON_PRODUCTION`.

---

## Thứ tự và cổng giữa các bước

| Mốc | Gồm | Điều kiện sang mốc sau |
|---|---|---|
| M1 | S0–S2 | Khớp crosswalk ≥ 95% và `wards.code` không lệch (nếu lệch: dừng, báo bạn) |
| M2 | S3–S6 | Có baseline; báo cáo clean cho thấy cỡ mẫu HCM-mới |
| M3 | S7–S9 | Gate đạt, artifact v0 xuất được |

## Việc cần bạn

1. Ghim `hf_revision` và `crosswalk_commit` (hoặc cho tôi tra khi làm S1).
2. Cấp cách lấy danh sách `wards.code` (kết nối replica hoặc CSV export) cho cổng ở S2.
3. Xác nhận 2 quyết định tôi tự đặt: ngoài khoảng → NaN thay vì kẹp (S3); phân vai tập early stopping / calibration (S7).
4. Test B cần quyền đọc replica; chưa có thì bỏ qua ở v0.
