---
name: obsidian-memory
description: Memory công việc dài hạn (kiến trúc, feature, bug, fix) ghi vào Obsidian vault ngoài repo tại D:\Code\V-nexus\v-nexus-extensions\.claude\memory. Thay thế memory-mirror cho loại project.
status: live
updated: 2026-10-28
metadata:
  type: reference
---

# Obsidian memory vault

Vault: `D:\Code\V-nexus\v-nexus-extensions\.claude\memory\` (mở thư mục này làm vault trong Obsidian). Cấu trúc + quy ước gốc: `README.md` và `_index.md` trong vault.

## Phân vai

- **Vault** = tri thức công việc dài hạn: `architecture/` (A-NNN), `features/` (F-NNN), `bugs/` (B-NNN), `fixes/` (X-NNN).
- **Auto-memory** (`~/.claude/projects/<hash>/memory/`) = vẫn là nơi cho `user` / `feedback` / `reference` và index `MEMORY.md` nạp mỗi session.
- Loại `project` KHÔNG mirror sang `.claude/memory-mirror/` nữa — ghi thẳng vào vault. Quy tắc mirror trong [[memory-mirror]] không áp dụng cho loại project khi vault khả dụng; `.claude/memory-mirror/` giữ nguyên làm lịch sử.

## Cách ghi

1. Chọn loại note → copy template `_templates/<type>.md` trong vault (frontmatter bắt buộc: `type`, `id`, `created`, `updated`, `status`, `tags`, `related`).
2. ID = số lớn nhất hiện có trong thư mục + 1. Tên file `<id>-<kebab-slug>.md`.
3. Thêm 1 dòng vào `<type>/_moc.md` của note đó; sửa `updated` trong frontmatter khi cập nhật.
4. Link chéo bằng wikilink (`[[bugs/B-007]]`); bug note link sang fix, không lặp nội dung fix.
5. Note ngắn, nhiều link. Không ghi secret/token/PII giá trị thật.

## Ranh giới

- Vault nằm ngoài repo này: note file `A-/F-/X-/B-*.md` bị gitignore ở repo `v-nexus-extensions`, chỉ cấu trúc được commit. Không commit hộ vault từ session này.
- Vault không khả dụng (thư mục không tồn tại) → báo user, KHÔNG tự tạo vault mới; tạm ghi vào auto-memory.
- `user-prefs/` trong vault là cá nhân, không đụng.
