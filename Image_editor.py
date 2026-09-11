import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import io
import webbrowser
from ultralytics import YOLO

# =========================================================
# APPEARANCE & THEME
# =========================================================

ctk.set_appearance_mode("Dark")            # همیشه حالت تاریک
ctk.set_default_color_theme("dark-blue")   # تم مدرن

# رنگ‌های سفارشی مدرن
PRIMARY       = "#6366F1"   # Indigo مدرن
PRIMARY_HOVER = "#4F46E5"
ACCENT        = "#22D3EE"
SUCCESS       = "#10B981"
DANGER        = "#EF4444"
CAMERA_BG     = "#0D9488"

# توضیحات فیلترها برای Tooltip
FILTER_TOOLTIPS = {
    "Blur": "Average blur — softens the image by averaging neighboring pixels.",
    "Gaussian": "Gaussian blur — smooths the image with a Gaussian kernel (softer edges).",
    "Sharpen": "Sharpens details and edges by enhancing contrast between pixels.",
    "Median": "Median blur — removes noise while preserving edges better than average blur.",
    "Bilateral": "Bilateral filter — smooths noise but keeps edges sharp.",
    "Flip": "Flips the image horizontally or vertically.",
    "Rotation": "Rotates the image by a chosen angle (degrees).",
    "Noise": "Adds random Gaussian noise to the image.",
    "Resize": "Changes the width and height of the image.",
}


class ToolTip:
    """Simple tooltip that appears on hover."""
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        label = ctk.CTkLabel(
            tw, text=self.text, justify="left",
            font=ctk.CTkFont(size=12),
            fg_color=("#1E1B4B", "#1E1B4B"),
            text_color="#E0E7FF",
            corner_radius=8,
            padx=12, pady=8
        )
        label.pack()

    def _hide(self, event=None):
        self._cancel()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# =========================================================
# APPLICATION
# =========================================================
import sys

def resource_path(relative_path):
    """مسیر درست فایل‌ها هم در حالت عادی و هم داخل exe"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ImageEditor(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Image Editor — Modern")
        self.geometry("1620x960")
        self.minsize(1320, 760)

        # ----- State -----
        self.original_image = None
        self.current_image = None
        self.annotated_image = None
        self.original_path = None
        self.original_size_bytes = 0

        self.mode = "filters"          # "filters" یا "detection"
        self.filters = {}
        self.settings = {
            "Blur": 3, "Gaussian": 3, "Sharpen": 1,
            "Median": 3, "Bilateral": 5, "Flip": "Horizontal",
            "Rotation": 0, "Noise": 0,
            "ResizeW": 800, "ResizeH": 600
        }
        self.active_order = []
        self.setting_cards = {}
        self.slider_widgets = {}
        self.resize_entries = {}

        self.history = []
        self.history_index = -1
        self.last_history_signature = None
        self.noise_seed = 12345
        self.expanded_groups = set()
        self._history_scroll_pos = 0.0

        # ----- Detection State -----
        self.yolo_model = None
        self.detection_results = None      # نتایج کامل YOLO
        self.active_detections = []        # لیست detectionهایی که فعلاً نمایش داده می‌شن
        self.deleted_detections = []       # اشیاء حذف‌شده

        self.create_menu()
        self.create_top_bar()
        self.create_main_layout()

        self.bind("<Configure>", self.on_window_resize)
        self.update_ui()

    # =====================================================
    # MENU
    # =====================================================

    def create_menu(self):
        # منوی ساده با tkinter معمولی (customtkinter منو نداره)
        import tkinter as tk
        menubar = tk.Menu(self, tearoff=0)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", accelerator="Ctrl+O", command=self.open_image)
        file_menu.add_command(label="Open Camera", accelerator="Ctrl+Shift+C", command=self.open_camera)
        file_menu.add_command(label="Save PNG", accelerator="Ctrl+S", command=lambda: self.save_image("PNG"))
        file_menu.add_command(label="Save JPG", command=lambda: self.save_image("JPEG"))
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.close_image)
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        edit_menu.add_command(label="Reset All", accelerator="Ctrl+R", command=self.reset_all)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="GitHub", command=lambda: webbrowser.open("https://github.com/mohammadmahdimoradi-git"))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menubar)

        self.bind_all("<Control-o>", lambda e: self.open_image())
        self.bind_all("<Control-O>", lambda e: self.open_image())
        self.bind_all("<Control-s>", lambda e: self.save_image("PNG"))
        self.bind_all("<Control-S>", lambda e: self.save_image("PNG"))
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-Z>", lambda e: self.undo())
        self.bind_all("<Control-y>", lambda e: self.redo())
        self.bind_all("<Control-Y>", lambda e: self.redo())
        self.bind_all("<Control-r>", lambda e: self.reset_all())
        self.bind_all("<Control-R>", lambda e: self.reset_all())
        self.bind_all("<Control-Shift-c>", lambda e: self.open_camera())
        self.bind_all("<Control-Shift-C>", lambda e: self.open_camera())

    # =====================================================
    # TOP BAR
    # =====================================================

    def create_top_bar(self):
        self.top_bar = ctk.CTkFrame(self, height=64, corner_radius=0, fg_color=("gray92", "gray14"))
        self.top_bar.pack(fill="x", padx=0, pady=0)
        self.top_bar.pack_propagate(False)

        # عنوان
        title = ctk.CTkLabel(self.top_bar, text="IMAGE EDITOR",
                             font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
                             text_color=PRIMARY)
        title.pack(side="left", padx=20, pady=14)

        # ----- سمت راست -----
        right = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        right.pack(side="right", padx=12)

        # Help
        ctk.CTkButton(right, text="Help", width=70, height=36, corner_radius=10,
                      fg_color="#475569", hover_color="#334155",
                      command=self.show_help).pack(side="right", padx=4, pady=14)

        # Reset
        self.reset_button = ctk.CTkButton(right, text="Reset", width=80, height=36,
                                          corner_radius=10, fg_color=DANGER, hover_color="#DC2626",
                                          command=self.reset_all)
        self.reset_button.pack(side="right", padx=4, pady=14)

        # Redo
        self.redo_button = ctk.CTkButton(right, text="Redo", width=70, height=36,
                                         corner_radius=10, command=self.redo)
        self.redo_button.pack(side="right", padx=4, pady=14)

        # Undo
        self.undo_button = ctk.CTkButton(right, text="Undo", width=70, height=36,
                                         corner_radius=10, command=self.undo)
        self.undo_button.pack(side="right", padx=4, pady=14)

        # Camera
        self.camera_button = ctk.CTkButton(right, text="📷  Camera", width=120, height=36,
                                           corner_radius=10, fg_color=CAMERA_BG, hover_color="#0F766E",
                                           command=self.open_camera)
        self.camera_button.pack(side="right", padx=4, pady=14)

        # Object Detection
        self.detection_button = ctk.CTkButton(
            right, text="🔍  Object Detection", width=160, height=36,
            corner_radius=10, fg_color="#7C3AED", hover_color="#6D28D9",
            command=self.toggle_detection_mode
        )
        self.detection_button.pack(side="right", padx=4, pady=14)

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    def create_main_layout(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=(8, 6))

        # ---------- 1. FILTERS COLUMN ----------
        self.filter_col = ctk.CTkFrame(main, width=168, corner_radius=14)
        self.filter_col.pack(side="left", fill="y", padx=(0, 10))
        self.filter_col.pack_propagate(False)

        ctk.CTkLabel(self.filter_col, text="FILTERS",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=PRIMARY).pack(anchor="w", padx=14, pady=(16, 12))

        self.filter_list = ctk.CTkFrame(self.filter_col, fg_color="transparent")
        self.filter_list.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        filters = ["Blur", "Gaussian", "Sharpen", "Median", "Bilateral",
                   "Flip", "Rotation", "Noise", "Resize"]

        for name in filters:
            btn = ctk.CTkButton(
                self.filter_list, text=name, height=42,
                corner_radius=10, font=ctk.CTkFont(size=13, weight="bold"),
                fg_color=("#E0E7FF", "#312E81"),
                text_color=(PRIMARY, "#C7D2FE"),
                hover_color=(PRIMARY, PRIMARY),
                anchor="w",
                command=lambda n=name: self.toggle_filter(n)
            )
            btn.pack(fill="x", pady=4)
            setattr(self, f"btn_{name}", btn)
            # توضیح هنگام نگه داشتن ماوس
            ToolTip(btn, FILTER_TOOLTIPS.get(name, name))

        # ---------- 2. INFO / DETECTION COLUMN ----------
        self.info_col = ctk.CTkFrame(main, width=320, corner_radius=14)
        self.info_col.pack(side="left", fill="y", padx=(0, 10))
        self.info_col.pack_propagate(False)

        # این فریم‌ها رو بعداً پر می‌کنیم
        self.filters_panel = ctk.CTkFrame(self.info_col, fg_color="transparent")
        self.detection_panel = ctk.CTkFrame(self.info_col, fg_color="transparent")

        self.build_filters_panel()
        self.build_detection_panel()

        self.filters_panel.pack(fill="both", expand=True)
        # detection_panel در ابتدا مخفی است

        # ---------- 3. BEFORE ----------
        self.before_col = ctk.CTkFrame(main, corner_radius=14)
        self.before_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(self.before_col, text="BEFORE",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=("gray50", "gray60")).pack(anchor="w", padx=16, pady=(14, 4))

        # مشخصات عکس اصلی — بالای تصویر
        orig_info_box = ctk.CTkFrame(self.before_col, corner_radius=10, fg_color=("#EEF2FF", "#1E1B4B"))
        orig_info_box.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(orig_info_box, text="ORIGINAL", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=PRIMARY).pack(anchor="w", padx=10, pady=(6, 1))
        self.original_info = ctk.CTkLabel(orig_info_box, text="No image", font=ctk.CTkFont(size=12),
                                          justify="left", anchor="w")
        self.original_info.pack(fill="x", padx=10, pady=(0, 6))

        self.orig_label = ctk.CTkLabel(self.before_col, text="Open an image or use Camera",
                                       font=ctk.CTkFont(size=15, weight="bold"),
                                       text_color=("gray55", "gray50"))
        self.orig_label.place(relx=0.5, rely=0.55, anchor="center")

        # ---------- 4. AFTER ----------
        self.after_col = ctk.CTkFrame(main, corner_radius=14)
        self.after_col.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(self.after_col, text="AFTER",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=("gray50", "gray60")).pack(anchor="w", padx=16, pady=(14, 4))

        # مشخصات عکس تغییر‌یافته — بالای تصویر
        mod_info_box = ctk.CTkFrame(self.after_col, corner_radius=10, fg_color=("#ECFDF5", "#064E3B"))
        mod_info_box.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(mod_info_box, text="MODIFIED", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=SUCCESS).pack(anchor="w", padx=10, pady=(6, 1))
        self.modified_info = ctk.CTkLabel(mod_info_box, text="No image", font=ctk.CTkFont(size=12),
                                          justify="left", anchor="w")
        self.modified_info.pack(fill="x", padx=10, pady=(0, 6))

        self.mod_label = ctk.CTkLabel(self.after_col, text="Changes appear here",
                                      font=ctk.CTkFont(size=15, weight="bold"),
                                      text_color=("gray55", "gray50"))
        self.mod_label.place(relx=0.5, rely=0.55, anchor="center")

    def build_filters_panel(self):
        # Active Filters — ارتفاعش با تعداد فیلترها تنظیم می‌شه
        ctk.CTkLabel(self.filters_panel, text="ACTIVE FILTERS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=14, pady=(12, 4))

        self.active_filters_frame = ctk.CTkFrame(
            self.filters_panel, height=36, corner_radius=8, fg_color="transparent"
        )
        self.active_filters_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.active_filters_frame.pack_propagate(False)

        # Settings (کمی کوتاه‌تر تا History فضای بیشتری بگیرد)
        ctk.CTkLabel(self.filters_panel, text="SETTINGS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=14, pady=(6, 4))

        self.settings_outer = ctk.CTkScrollableFrame(self.filters_panel, height=120, corner_radius=10)
        self.settings_outer.pack(fill="x", padx=10, pady=(0, 8))

        # History — فضای باقی‌مانده را می‌گیرد (مشخصات عکس به بالای تصاویر منتقل شد)
        hist_head = ctk.CTkFrame(self.filters_panel, fg_color="transparent")
        hist_head.pack(fill="x", padx=12, pady=(12, 4))
        ctk.CTkLabel(hist_head, text="HISTORY",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=ACCENT).pack(side="left")

        self.history_scrollable = ctk.CTkScrollableFrame(self.filters_panel, corner_radius=10)
        self.history_scrollable.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def build_detection_panel(self):
        ctk.CTkLabel(self.detection_panel, text="OBJECT DETECTION",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#A78BFA").pack(anchor="w", padx=14, pady=(16, 10))

        # دکمه Detect
        self.detect_btn = ctk.CTkButton(
            self.detection_panel, text="▶  Detect Objects", height=42,
            corner_radius=12, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#7C3AED", hover_color="#6D28D9",
            command=self.run_detection
        )
        self.detect_btn.pack(fill="x", padx=14, pady=(0, 12))

        # اطلاعات خلاصه
        self.detection_summary = ctk.CTkLabel(
            self.detection_panel, text="No detection yet",
            font=ctk.CTkFont(size=13), text_color=("gray50", "gray60")
        )
        self.detection_summary.pack(anchor="w", padx=16, pady=(0, 8))

        # لیست اشیاء
        ctk.CTkLabel(self.detection_panel, text="DETECTED OBJECTS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=14, pady=(8, 4))

        self.detection_list_frame = ctk.CTkScrollableFrame(self.detection_panel, corner_radius=10)
        self.detection_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # دکمه Reset
        self.reset_det_btn = ctk.CTkButton(
            self.detection_panel, text="↺  Reset Deleted Objects", height=38,
            corner_radius=10, fg_color="#475569", hover_color="#334155",
            command=self.reset_deleted_detections
        )
        self.reset_det_btn.pack(fill="x", padx=14, pady=(0, 14))

    # =====================================================
    # TOGGLE DETECTION MODE
    # =====================================================

    def toggle_detection_mode(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please open an image first.")
            return

        if self.mode == "filters":
            self.mode = "detection"
            self.filters_panel.pack_forget()
            self.detection_panel.pack(fill="both", expand=True)
            self.detection_button.configure(fg_color="#4C1D95", text="🔍  Detection ON")
            # اگر قبلاً تشخیص نداده، خودکار اجرا کن
            if self.detection_results is None:
                self.run_detection()
        else:
            self.mode = "filters"
            self.detection_panel.pack_forget()
            self.filters_panel.pack(fill="both", expand=True)
            self.detection_button.configure(fg_color="#7C3AED", text="🔍  Object Detection")
            self.annotated_image = None
            self.update_image()

    # =====================================================
    # YOLO DETECTION
    # =====================================================

    def load_yolo_model(self):
        if self.yolo_model is not None:
            return True
        try:
            # می‌تونی مسیر مدل رو اینجا تغییر بدی
            self.yolo_model = YOLO(resource_path("yolov8n.pt"))   # یا "Model/yolo12l.pt"
            return True
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load YOLO model:\n{e}")
            return False

    def run_detection(self):
        if self.original_image is None:
            return
        if not self.load_yolo_model():
            return

        self.update()

        try:
            # تبدیل به فرمت OpenCV
            img = np.array(self.current_image if self.current_image else self.original_image)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            results = self.yolo_model.predict(img_bgr, conf=0.35, verbose=False)
            self.detection_results = results[0]

            # ساخت لیست detectionها
            self.active_detections = []
            self.deleted_detections = []

            if self.detection_results.boxes is not None:
                for i, box in enumerate(self.detection_results.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.detection_results.names[cls_id]
                    self.active_detections.append({
                        "index": i,
                        "name": name,
                        "conf": conf,
                        "box": box
                    })

            self.update_detection_ui()
            self.draw_detections()

        except Exception as e:
            messagebox.showerror("Detection Error", str(e))

    def draw_detections(self):
        if self.detection_results is None:
            return

        # کپی از تصویر اصلی
        img = np.array(self.current_image if self.current_image else self.original_image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # فقط detectionهای فعال رو رسم کن
        for det in self.active_detections:
            box = det["box"]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = det["conf"]
            name = det["name"]

            # رنگ تصادفی اما ثابت بر اساس نام کلاس
            color = self._get_color_for_class(name)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.annotated_image = Image.fromarray(rgb)
        self.update_image()

    def _get_color_for_class(self, name):
        # رنگ ثابت بر اساس نام کلاس
        colors = {
            "person": (255, 99, 71), "car": (30, 144, 255), "truck": (255, 165, 0),
            "bus": (50, 205, 50), "motorcycle": (255, 20, 147), "bicycle": (0, 206, 209),
            "dog": (255, 215, 0), "cat": (186, 85, 211), "chair": (70, 130, 180),
            "bottle": (255, 105, 180), "cup": (0, 191, 255), "laptop": (147, 112, 219)
        }
        return colors.get(name, (100, 149, 237))

    def update_detection_ui(self):
        # پاک کردن لیست قبلی
        for w in self.detection_list_frame.winfo_children():
            w.destroy()

        if not self.active_detections:
            ctk.CTkLabel(self.detection_list_frame, text="No objects detected",
                         text_color=("gray50", "gray60")).pack(pady=20)
            self.detection_summary.configure(text="0 objects")
            return

        # شمارش کلاس‌ها
        from collections import Counter
        counts = Counter([d["name"] for d in self.active_detections])
        summary_text = "  •  ".join([f"{k}: {v}" for k, v in counts.most_common()])
        self.detection_summary.configure(text=summary_text)

        # نمایش هر آبجکت
        for det in self.active_detections:
            card = ctk.CTkFrame(self.detection_list_frame, corner_radius=8, height=42)
            card.pack(fill="x", pady=3, padx=2)
            card.pack_propagate(False)

            label = ctk.CTkLabel(
                card,
                text=f"  {det['name']}   ({det['conf']:.2f})",
                font=ctk.CTkFont(size=13),
                anchor="w"
            )
            label.pack(side="left", fill="x", expand=True, padx=(8, 0))

            # دکمه حذف
            del_btn = ctk.CTkButton(
                card, text="✕", width=32, height=28,
                corner_radius=8, fg_color="transparent",
                hover_color=DANGER, text_color=DANGER,
                font=ctk.CTkFont(size=14, weight="bold"),
                command=lambda d=det: self.delete_detection(d)
            )
            del_btn.pack(side="right", padx=6)

    def delete_detection(self, det):
        if det in self.active_detections:
            self.active_detections.remove(det)
            self.deleted_detections.append(det)
            self.update_detection_ui()
            self.draw_detections()

    def reset_deleted_detections(self):
        if not self.deleted_detections:
            return
        self.active_detections.extend(self.deleted_detections)
        self.deleted_detections.clear()
        # مرتب کردن دوباره بر اساس ایندکس اصلی
        self.active_detections.sort(key=lambda x: x["index"])
        self.update_detection_ui()
        self.draw_detections()

    # =====================================================
    # HELP
    # =====================================================

    def show_help(self):
        win = ctk.CTkToplevel(self)
        win.title("User Guide")
        win.geometry("700x620")
        win.transient(self)

        text = ctk.CTkTextbox(win, font=ctk.CTkFont(size=13), wrap="word")
        text.pack(fill="both", expand=True, padx=16, pady=(16, 8))

        guide = """
IMAGE EDITOR — User Guide

1. File Operations
• Open Image (Ctrl+O)
• Open Camera (Ctrl+Shift+C)
• Save PNG / JPG

2. Filters
Click any filter on the left to enable it.
Hover over a filter to see a short description.
Adjust values with the modern sliders.

3. Object Detection
• Click the "Object Detection" button next to Camera.
• The middle panel switches to Detection mode.
• Click "Detect Objects" to run YOLO.
• Detected objects appear in a list.
• Click the ✕ button next to any object to remove it from the image.
• Use "Reset Deleted Objects" to bring them back.

4. History
Every real change is saved. Click any history item to restore that state.

5. GitHub
Click the button below to open the project on GitHub.
"""
        text.insert("1.0", guide)
        text.configure(state="disabled")

        # دکمه لینک GitHub — با کلیک باز می‌شود
        github_btn = ctk.CTkButton(
            win,
            text="🔗  https://github.com/mohammadmahdimoradi-git",
            height=40,
            corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7C3AED",
            hover_color="#6D28D9",
            command=lambda: webbrowser.open("https://github.com/mohammadmahdimoradi-git")
        )
        github_btn.pack(fill="x", padx=16, pady=(0, 16))

    # =====================================================
    # OPEN / CLOSE / RESET  (بخش‌های اصلی بدون تغییر زیاد)
    # =====================================================

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return
        self.update()
        captured = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(display, "SPACE = Capture    Q / Esc = Cancel",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Camera - Press SPACE to capture', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == 32:
                captured = frame.copy()
                break
            elif key == ord('q') or key == ord('Q') or key == 27:
                break
        cv2.destroyAllWindows()
        cap.release()
        if captured is None:
            return
        try:
            rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self._load_image(image, "Camera Capture", 0)
        except Exception as e:
            messagebox.showerror("Capture Error", str(e))

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            self._load_image(image, path, os.path.getsize(path))
        except Exception as e:
            messagebox.showerror("Open Error", str(e))

    def _load_image(self, image, path, size_bytes):
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.annotated_image = None
        self.original_path = path
        self.original_size_bytes = size_bytes
        w, h = image.size

        self.filters.clear()
        self.active_order.clear()
        self.settings = {
            "Blur": 3, "Gaussian": 3, "Sharpen": 1, "Median": 3, "Bilateral": 5,
            "Flip": "Horizontal", "Rotation": 0, "Noise": 0,
            "ResizeW": w, "ResizeH": h
        }
        self.history.clear()
        self.history_index = -1
        self.last_history_signature = None
        self.expanded_groups.clear()

        # ریست detection
        self.detection_results = None
        self.active_detections = []
        self.deleted_detections = []

        self.clear_settings()
        name = "Original Image (Camera)" if path == "Camera Capture" else "Original Image"
        self.add_history(name)
        self.update_image()
        self.update_ui()

        # اگر در حالت detection بودیم، برگرد به فیلتر
        if self.mode == "detection":
            self.toggle_detection_mode()

    def close_image(self):
        self.original_image = self.current_image = self.annotated_image = None
        self.original_path = None
        self.original_size_bytes = 0
        self.filters.clear()
        self.active_order.clear()
        self.history.clear()
        self.history_index = -1
        self.detection_results = None
        self.active_detections = []
        self.deleted_detections = []
        self.clear_settings()
        self.orig_label.configure(image=None, text="Open an image or use Camera")
        self.mod_label.configure(image=None, text="Changes appear here")
        self.update_ui()

    def reset_all(self):
        if self.original_image is None:
            return
        w, h = self.original_image.size
        self.filters.clear()
        self.active_order.clear()
        self.settings = {
            "Blur": 3, "Gaussian": 3, "Sharpen": 1, "Median": 3, "Bilateral": 5,
            "Flip": "Horizontal", "Rotation": 0, "Noise": 0,
            "ResizeW": w, "ResizeH": h
        }
        self.current_image = self.original_image.copy()
        self.annotated_image = None
        self.history.clear()
        self.history_index = -1
        self.last_history_signature = None
        self.detection_results = None
        self.active_detections = []
        self.deleted_detections = []
        self.clear_settings()
        self.add_history("Original Image")
        self.update_image()
        self.update_ui()

    # =====================================================
    # FILTER LOGIC (تقریباً همان منطق قبلی)
    # =====================================================

    def toggle_filter(self, name):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please open an image first.")
            return
        if self.mode == "detection":
            messagebox.showinfo("Mode", "Please switch back to Filters mode first.")
            return

        if self.filters.get(name, False):
            self.filters[name] = False
            if name in self.active_order:
                self.active_order.remove(name)
            self.hide_setting_card(name)
            self.update_filter_button(name)
            self.apply_filters(add_history=True)
        else:
            self.filters[name] = True
            if name not in self.active_order:
                self.active_order.append(name)
            self.show_setting_card(name)
            self.update_filter_button(name)
            self.apply_filters(add_history=False)

    def update_filter_button(self, name):
        btn = getattr(self, f"btn_{name}")
        if self.filters.get(name, False):
            btn.configure(fg_color=PRIMARY, text_color="white")
        else:
            btn.configure(fg_color=("#E0E7FF", "#312E81"),
                          text_color=(PRIMARY, "#C7D2FE"))

    def show_setting_card(self, name):
        if name in self.setting_cards:
            return

        card = ctk.CTkFrame(self.settings_outer, corner_radius=8)
        card.pack(fill="x", pady=4, padx=2)

        ctk.CTkLabel(card, text=name, width=80, anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=8)

        if name == "Flip":
            var = ctk.StringVar(value=self.settings["Flip"])
            menu = ctk.CTkOptionMenu(card, values=["Horizontal", "Vertical"],
                                     variable=var, width=110,
                                     command=lambda v: self.setting_changed(name, v, finished=True))
            menu.pack(side="left", padx=6, pady=6)
            self.setting_cards[name] = card
            return

        if name == "Resize":
            cont = ctk.CTkFrame(card, fg_color="transparent")
            cont.pack(side="left", fill="x", expand=True, padx=4, pady=6)

            w_e = ctk.CTkEntry(cont, width=55, justify="center")
            w_e.insert(0, str(self.settings["ResizeW"]))
            w_e.pack(side="left", padx=2)

            h_e = ctk.CTkEntry(cont, width=55, justify="center")
            h_e.insert(0, str(self.settings["ResizeH"]))
            h_e.pack(side="left", padx=2)

            def apply_r():
                try:
                    self.settings["ResizeW"] = max(1, int(w_e.get()))
                    self.settings["ResizeH"] = max(1, int(h_e.get()))
                    self.apply_filters(add_history=True)
                except ValueError:
                    pass

            def reset_s():
                if self.original_image is None:
                    return
                ow, oh = self.original_image.size
                self.settings["ResizeW"], self.settings["ResizeH"] = ow, oh
                w_e.delete(0, "end")
                w_e.insert(0, str(ow))
                h_e.delete(0, "end")
                h_e.insert(0, str(oh))
                self.apply_filters(add_history=True)

            ctk.CTkButton(cont, text="Apply", width=55, height=28,
                          command=apply_r).pack(side="left", padx=3)
            ctk.CTkButton(cont, text="Reset", width=55, height=28,
                          fg_color=DANGER, command=reset_s).pack(side="left")
            self.resize_entries[name] = (w_e, h_e)
            self.setting_cards[name] = card
            return

        ranges = {
            "Blur": (1, 15), "Gaussian": (1, 15), "Sharpen": (0, 5),
            "Median": (1, 15), "Bilateral": (1, 15),
            "Rotation": (-180, 180), "Noise": (0, 100)
        }
        mn, mx = ranges[name]

        val_lbl = ctk.CTkLabel(card, text=str(self.settings[name]), width=40)
        val_lbl.pack(side="right", padx=8)

        slider = ctk.CTkSlider(
            card, from_=mn, to=mx, number_of_steps=int(mx - mn) if mx - mn <= 50 else 100,
            width=130,
            command=lambda v, n=name, l=val_lbl: self.setting_changed(n, v, l, finished=False)
        )
        slider.set(self.settings[name])
        slider.pack(side="left", fill="x", expand=True, padx=6, pady=8)

        # وقتی رها شد، به تاریخچه اضافه کن
        slider.bind("<ButtonRelease-1>",
                    lambda e, n=name: self.setting_changed(n, slider.get(), finished=True))

        self.slider_widgets[name] = slider
        self.setting_cards[name] = card

    def hide_setting_card(self, name):
        if name in self.setting_cards:
            self.setting_cards[name].destroy()
            del self.setting_cards[name]

    def clear_settings(self):
        for w in self.settings_outer.winfo_children():
            w.destroy()
        self.setting_cards.clear()
        self.slider_widgets.clear()
        self.resize_entries.clear()

    def setting_changed(self, name, value, label=None, finished=False):
        if name == "Flip":
            self.settings[name] = value
        else:
            num = float(value)
            self.settings[name] = int(num) if num.is_integer() else round(num, 1)
            if label:
                label.configure(text=str(self.settings[name]))
        self.apply_filters(add_history=False, light_update=True)
        if finished:
            self.add_history(self.get_history_name())
            self.update_ui()

    def apply_filters(self, add_history=True, light_update=False):
        if self.original_image is None:
            return
        try:
            img = np.array(self.original_image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for name in self.active_order:
                if not self.filters.get(name, False):
                    continue
                if name == "Blur":
                    k = int(self.settings["Blur"])
                    if k % 2 == 0: k += 1
                    img = cv2.blur(img, (k, k))
                elif name == "Gaussian":
                    k = int(self.settings["Gaussian"])
                    if k % 2 == 0: k += 1
                    img = cv2.GaussianBlur(img, (k, k), 0)
                elif name == "Sharpen":
                    a = float(self.settings["Sharpen"])
                    if a > 0:
                        img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5 + a, -1], [0, -1, 0]]))
                elif name == "Median":
                    k = int(self.settings["Median"])
                    if k % 2 == 0: k += 1
                    img = cv2.medianBlur(img, k)
                elif name == "Bilateral":
                    img = cv2.bilateralFilter(img, int(self.settings["Bilateral"]), 75, 75)
                elif name == "Flip":
                    img = cv2.flip(img, 1 if self.settings["Flip"] == "Horizontal" else 0)
                elif name == "Rotation":
                    ang = float(self.settings["Rotation"])
                    if ang != 0:
                        h, w = img.shape[:2]
                        m = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
                        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                elif name == "Noise":
                    amt = float(self.settings["Noise"])
                    if amt > 0:
                        rng = np.random.default_rng(self.noise_seed)
                        img = np.clip(img.astype(np.float32) + rng.normal(0, amt, img.shape), 0, 255).astype(np.uint8)
                elif name == "Resize":
                    img = cv2.resize(img, (max(1, int(self.settings["ResizeW"])),
                                           max(1, int(self.settings["ResizeH"]))),
                                     interpolation=cv2.INTER_AREA)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(img)
            self.annotated_image = None   # بعد از فیلتر، annotation باطل میشه

            if add_history:
                self.add_history(self.get_history_name())
            self.update_image()
            if not light_update:
                self.update_ui()
            else:
                self._update_image_details()
        except Exception as e:
            print("Filter Error:", e)

    # =====================================================
    # HISTORY
    # =====================================================

    def get_history_name(self):
        active = [n for n in self.active_order if self.filters.get(n, False)]
        return " + ".join(active) if active else "No Filters"

    def get_settings_summary(self, settings, active_order, filters):
        parts = []
        for name in active_order:
            if not filters.get(name, False):
                continue
            if name == "Flip":
                parts.append(f"Flip={settings.get('Flip')}")
            elif name == "Resize":
                parts.append(f"{settings.get('ResizeW')}×{settings.get('ResizeH')}")
            else:
                parts.append(f"{name}={settings.get(name)}")
        return " • ".join(parts) if parts else "—"

    def add_history(self, name):
        if self.current_image is None or name == "No Filters":
            return
        img_copy = self.current_image.copy()
        sig = (tuple(self.active_order), tuple(sorted(self.filters.items())),
               tuple(sorted(self.settings.items())), img_copy.size)
        if sig == self.last_history_signature:
            return
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append({
            "image": img_copy, "name": name, "signature": sig,
            "filters": self.filters.copy(), "active_order": self.active_order.copy(),
            "settings": self.settings.copy(),
            "summary": self.get_settings_summary(self.settings, self.active_order, self.filters)
        })
        self.history_index = len(self.history) - 1
        self.last_history_signature = sig
        if len(self.history) > 100:
            self.history.pop(0)
            self.history_index -= 1

    def restore_history(self, index):
        if index < 0 or index >= len(self.history):
            return
        item = self.history[index]
        self.history_index = index
        self.current_image = item["image"].copy()
        self.filters = item["filters"].copy()
        self.active_order = item["active_order"].copy()
        self.settings = item["settings"].copy()
        self.annotated_image = None
        self.clear_settings()
        for name in self.active_order:
            if self.filters.get(name, False):
                self.show_setting_card(name)
        self.update_image()
        self.update_ui()

    def undo(self):
        if self.history_index > 0:
            self.restore_history(self.history_index - 1)

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.restore_history(self.history_index + 1)

    def build_history_groups(self):
        if not self.history:
            return []
        groups = [{"name": self.history[0]["name"], "items": [0]}]
        for i in range(1, len(self.history)):
            if self.history[i]["name"] == groups[-1]["name"]:
                groups[-1]["items"].append(i)
            else:
                groups.append({"name": self.history[i]["name"], "items": [i]})
        return groups

    def toggle_group(self, key):
        if key in self.expanded_groups:
            self.expanded_groups.remove(key)
        else:
            self.expanded_groups.add(key)
        self.update_ui()

    # =====================================================
    # UPDATE IMAGE & UI
    # =====================================================

    def update_image(self):
        if self.original_image is None:
            return

        self.before_col.update_idletasks()
        self.after_col.update_idletasks()
        # فضای بالای هر ستون برای عنوان + کادر مشخصات رزرو شده
        bw = max(220, self.before_col.winfo_width() - 30)
        bh = max(180, self.before_col.winfo_height() - 110)
        aw = max(220, self.after_col.winfo_width() - 30)
        ah = max(180, self.after_col.winfo_height() - 110)

        # Before
        orig = self.original_image.copy()
        orig.thumbnail((bw, bh), Image.Resampling.LANCZOS)
        self.tk_orig = ImageTk.PhotoImage(orig)
        self.orig_label.configure(image=self.tk_orig, text="")
        self.orig_label.place(relx=0.5, rely=0.58, anchor="center")

        # After
        display_img = self.annotated_image if self.annotated_image is not None else self.current_image
        if display_img:
            mod = display_img.copy()
            mod.thumbnail((aw, ah), Image.Resampling.LANCZOS)
            self.tk_mod = ImageTk.PhotoImage(mod)
            self.mod_label.configure(image=self.tk_mod, text="")
            self.mod_label.place(relx=0.5, rely=0.58, anchor="center")

    def _update_image_details(self):
        if not self.original_image:
            self.original_info.configure(text="No image")
            self.modified_info.configure(text="No image")
            return
        ow, oh = self.original_image.size
        kb = self.original_size_bytes / 1024 if self.original_size_bytes else 0
        fmt = self.original_image.format or ("Camera" if self.original_path == "Camera Capture" else "—")
        self.original_info.configure(text=f"{ow} × {oh}\n{fmt}\n{kb:.1f} KB" if kb else f"{ow} × {oh}\n{fmt}")

        if self.current_image:
            mw, mh = self.current_image.size
            png = self.get_encoded_size("PNG") / 1024
            jpg = self.get_encoded_size("JPEG") / 1024
            self.modified_info.configure(text=f"{mw} × {mh}\nPNG {png:.1f} KB\nJPG {jpg:.1f} KB")
        else:
            self.modified_info.configure(text="No image")

    def update_ui(self):
        # Active filters — ارتفاع فریم با تعداد ردیف‌ها تنظیم می‌شود
        for c in self.active_filters_frame.winfo_children():
            c.destroy()
        active = [n for n in self.active_order if self.filters.get(n, False)]
        if not active:
            ctk.CTkLabel(self.active_filters_frame, text="None",
                         text_color=("gray50", "gray60")).pack(anchor="w", pady=4)
            self.active_filters_frame.configure(height=36)
        else:
            cols_per_row = 3
            num_rows = (len(active) + cols_per_row - 1) // cols_per_row
            # هر ردیف حدود 32px + کمی فاصله
            self.active_filters_frame.configure(height=max(36, num_rows * 34 + 4))
            for i in range(0, len(active), cols_per_row):
                row = ctk.CTkFrame(self.active_filters_frame, fg_color="transparent")
                row.pack(fill="x", pady=1)
                for name in active[i:i + cols_per_row]:
                    chip = ctk.CTkLabel(row, text=f"  {name}  ",
                                        font=ctk.CTkFont(size=11, weight="bold"),
                                        fg_color=("#E0E7FF", "#312E81"),
                                        text_color=(PRIMARY, "#C7D2FE"),
                                        corner_radius=6)
                    chip.pack(side="left", padx=3, pady=2)

        self._update_image_details()

        # History
        for c in self.history_scrollable.winfo_children():
            c.destroy()

        for g_idx, group in enumerate(self.build_history_groups()):
            key = f"{group['name']}_{g_idx}"
            is_exp = key in self.expanded_groups
            is_current_group = any(i == self.history_index for i in group["items"])

            header = ctk.CTkFrame(self.history_scrollable, corner_radius=10,
                                  fg_color=("#E0E7FF", "#1E1B4B") if is_current_group else ("gray90", "gray20"))
            header.pack(fill="x", pady=4)

            arrow = "▼" if is_exp else "▶"
            lbl = ctk.CTkLabel(header, text=f"  {arrow}  {group['name']}  ({len(group['items'])})",
                               font=ctk.CTkFont(size=13, weight="bold"),
                               anchor="w", cursor="hand2")
            lbl.pack(fill="x", padx=10, pady=10)
            lbl.bind("<Button-1>", lambda e, k=key: self.toggle_group(k))
            header.bind("<Button-1>", lambda e, k=key: self.toggle_group(k))

            if is_exp:
                for hist_idx in group["items"]:
                    item = self.history[hist_idx]
                    is_cur = hist_idx == self.history_index
                    child = ctk.CTkFrame(self.history_scrollable, corner_radius=8,
                                         fg_color=("#DBEAFE", "#312E81") if is_cur else ("gray95", "gray17"))
                    child.pack(fill="x", padx=(14, 4), pady=2)

                    prefix = "●  " if is_cur else "○  "
                    cl = ctk.CTkLabel(child, text=prefix + item.get("summary", "—"),
                                      font=ctk.CTkFont(size=13),
                                      anchor="w", cursor="hand2")
                    cl.pack(fill="x", padx=12, pady=9)
                    cl.bind("<Button-1>", lambda e, i=hist_idx: self.restore_history(i))
                    child.bind("<Button-1>", lambda e, i=hist_idx: self.restore_history(i))

        # دکمه‌های Undo / Redo / Reset
        self.undo_button.configure(state="normal" if self.history_index > 0 else "disabled")
        self.redo_button.configure(state="normal" if self.history_index < len(self.history) - 1 else "disabled")
        self.reset_button.configure(state="normal" if self.original_image else "disabled")

        for name in ["Blur", "Gaussian", "Sharpen", "Median", "Bilateral",
                     "Flip", "Rotation", "Noise", "Resize"]:
            self.update_filter_button(name)

    def get_encoded_size(self, fmt):
        if self.current_image is None:
            return 0
        buf = io.BytesIO()
        if fmt == "JPEG":
            self.current_image.convert("RGB").save(buf, format="JPEG", quality=95)
        else:
            self.current_image.save(buf, format="PNG")
        return len(buf.getvalue())

    def save_image(self, fmt="PNG"):
        img_to_save = self.annotated_image if self.annotated_image is not None else self.current_image
        if img_to_save is None:
            messagebox.showinfo("No Image", "Please open an image first.")
            return
        ext = ".png" if fmt == "PNG" else ".jpg"
        types = [("PNG", "*.png")] if fmt == "PNG" else [("JPEG", "*.jpg")]
        path = filedialog.asksaveasfilename(title="Save Image", defaultextension=ext, filetypes=types)
        if not path:
            return
        try:
            if fmt == "JPEG":
                img_to_save.convert("RGB").save(path, "JPEG", quality=95)
            else:
                img_to_save.save(path, "PNG")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def on_window_resize(self, event):
        if event.widget == self:
            self.after(80, self.update_image)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app = ImageEditor()
    app.mainloop()