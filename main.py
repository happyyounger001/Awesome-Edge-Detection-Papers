import json
import random
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

APP_TITLE = "儿童刷牙监督程序"
WINDOW_SIZE = "720x640"
TOTAL_SECONDS = 120
SEGMENT_SECONDS = 30
REGIONS = ["左上牙", "右上牙", "左下牙", "右下牙"]
ENCOURAGEMENTS = [
    "你今天刷得真认真！",
    "牙齿亮晶晶！",
    "坚持刷牙真棒！",
    "太棒啦，又完成一次！",
]
RUNNING_MESSAGES = [
    "准备好了吗？开始刷牙吧！",
    "换一个区域继续刷！",
    "快完成啦，加油！",
]
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "brush_log.json"


class BrushStorage:
    """Handle local JSON storage for completed brushing sessions."""

    def __init__(self, data_file: Path) -> None:
        self.data_file = data_file
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self._write({"records": {}})

    def _read(self) -> dict:
        try:
            with self.data_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict) or "records" not in data:
                raise ValueError("Invalid storage format")
            return data
        except (json.JSONDecodeError, OSError, ValueError):
            data = {"records": {}}
            self._write(data)
            return data

    def _write(self, data: dict) -> None:
        with self.data_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def today_count(self) -> int:
        data = self._read()
        return int(data["records"].get(self._today_key(), 0))

    def add_completed_session(self) -> int:
        data = self._read()
        today = self._today_key()
        data["records"][today] = int(data["records"].get(today, 0)) + 1
        self._write(data)
        return int(data["records"][today])

    @staticmethod
    def _today_key() -> str:
        return datetime.now().strftime("%Y-%m-%d")


class KidBrushApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.storage = BrushStorage(DATA_FILE)
        self.remaining_seconds = TOTAL_SECONDS
        self.is_running = False
        self.timer_job: str | None = None
        self.completed_today = self.storage.today_count()
        self.last_region_index = -1

        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(640, 580)
        self.root.configure(bg="#FFF6D6")

        self.title_var = tk.StringVar(value="准备好了吗？开始刷牙吧！")
        self.timer_var = tk.StringVar(value=self.format_time(TOTAL_SECONDS))
        self.region_var = tk.StringVar(value="当前区域：左上牙")
        self.status_var = tk.StringVar(value="当前状态：等待开始")
        self.progress_var = tk.StringVar(value="进度：0%")
        self.today_var = tk.StringVar(value=f"今天已经刷牙 {self.completed_today} 次")
        self.star_var = tk.StringVar(value="⭐ ⭐ ⭐")

        self._build_ui()
        self._refresh_display()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#FFF6D6", padx=24, pady=20)
        container.pack(fill="both", expand=True)

        title = tk.Label(
            container,
            text=APP_TITLE,
            font=("Arial", 28, "bold"),
            bg="#FFF6D6",
            fg="#FF7A59",
        )
        title.pack(pady=(0, 12))

        subtitle = tk.Label(
            container,
            textvariable=self.title_var,
            font=("Arial", 18, "bold"),
            bg="#FFF6D6",
            fg="#4E5B7C",
            wraplength=620,
        )
        subtitle.pack(pady=(0, 20))

        timer_card = tk.Frame(container, bg="#FFFFFF", bd=0, highlightthickness=0)
        timer_card.pack(fill="x", pady=10)

        timer_label = tk.Label(
            timer_card,
            textvariable=self.timer_var,
            font=("Arial", 54, "bold"),
            bg="#FFFFFF",
            fg="#FF5E7E",
            pady=20,
        )
        timer_label.pack()

        info_frame = tk.Frame(container, bg="#FFF6D6")
        info_frame.pack(fill="x", pady=16)

        self._make_info_chip(info_frame, self.region_var, "#C9F9FF", "#006D77").pack(fill="x", pady=6)
        self._make_info_chip(info_frame, self.status_var, "#DFF8CC", "#2B9348").pack(fill="x", pady=6)
        self._make_info_chip(info_frame, self.progress_var, "#FFE4B5", "#C76D00").pack(fill="x", pady=6)

        progress_frame = tk.Frame(container, bg="#FFF6D6")
        progress_frame.pack(fill="x", pady=16)

        self.progress_canvas = tk.Canvas(
            progress_frame,
            height=36,
            bg="#FFF6D6",
            highlightthickness=0,
        )
        self.progress_canvas.pack(fill="x")

        button_frame = tk.Frame(container, bg="#FFF6D6")
        button_frame.pack(pady=20)

        self.start_button = tk.Button(
            button_frame,
            text="开始",
            command=self.start_or_resume,
            font=("Arial", 18, "bold"),
            bg="#7AD66D",
            fg="#FFFFFF",
            activebackground="#68C25D",
            width=10,
            pady=10,
            relief="flat",
            cursor="hand2",
        )
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        pause_button = tk.Button(
            button_frame,
            text="暂停",
            command=self.pause_timer,
            font=("Arial", 18, "bold"),
            bg="#FFB84D",
            fg="#FFFFFF",
            activebackground="#F4A62D",
            width=10,
            pady=10,
            relief="flat",
            cursor="hand2",
        )
        pause_button.grid(row=0, column=1, padx=10, pady=10)

        reset_button = tk.Button(
            button_frame,
            text="重置",
            command=self.reset_timer,
            font=("Arial", 18, "bold"),
            bg="#6AA9FF",
            fg="#FFFFFF",
            activebackground="#4F96F4",
            width=10,
            pady=10,
            relief="flat",
            cursor="hand2",
        )
        reset_button.grid(row=0, column=2, padx=10, pady=10)

        footer = tk.Frame(container, bg="#FFF0F5", padx=20, pady=16)
        footer.pack(fill="x", side="bottom", pady=(20, 0))

        today_label = tk.Label(
            footer,
            textvariable=self.today_var,
            font=("Arial", 18, "bold"),
            bg="#FFF0F5",
            fg="#7B2CBF",
        )
        today_label.pack(pady=6)

        stars_label = tk.Label(
            footer,
            textvariable=self.star_var,
            font=("Arial", 22, "bold"),
            bg="#FFF0F5",
            fg="#F4A261",
        )
        stars_label.pack(pady=6)

    def _make_info_chip(self, parent: tk.Widget, variable: tk.StringVar, bg: str, fg: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=bg, padx=18, pady=10)
        label = tk.Label(frame, textvariable=variable, font=("Arial", 18, "bold"), bg=bg, fg=fg)
        label.pack(anchor="center")
        return frame

    @staticmethod
    def format_time(total_seconds: int) -> str:
        minutes, seconds = divmod(max(total_seconds, 0), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def start_or_resume(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.status_var.set("当前状态：正在刷牙")
        self.title_var.set("准备好了吗？开始刷牙吧！" if self.remaining_seconds == TOTAL_SECONDS else "继续保持，刷得真棒！")
        self.start_button.config(text="继续")
        self._schedule_tick()

    def pause_timer(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.status_var.set("当前状态：已暂停")
        self.title_var.set("休息一下，准备好再继续刷牙！")

    def reset_timer(self) -> None:
        self.is_running = False
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.remaining_seconds = TOTAL_SECONDS
        self.last_region_index = -1
        self.status_var.set("当前状态：等待开始")
        self.title_var.set("准备好了吗？开始刷牙吧！")
        self.start_button.config(text="开始")
        self.star_var.set("⭐ ⭐ ⭐")
        self._refresh_display()

    def _schedule_tick(self) -> None:
        self.timer_job = self.root.after(1000, self._tick)

    def _tick(self) -> None:
        if not self.is_running:
            return
        self.remaining_seconds -= 1
        self._refresh_display()

        if self.remaining_seconds <= 0:
            self._complete_session()
            return

        self._schedule_tick()

    def _refresh_display(self) -> None:
        self.timer_var.set(self.format_time(self.remaining_seconds))
        current_region = self._current_region()
        self.region_var.set(f"当前区域：{current_region}")
        progress = int(((TOTAL_SECONDS - self.remaining_seconds) / TOTAL_SECONDS) * 100)
        self.progress_var.set(f"进度：{progress}%")
        self._update_message_for_region(progress)
        self._draw_progress_bar(progress)

    def _current_region(self) -> str:
        elapsed = TOTAL_SECONDS - self.remaining_seconds
        index = min(elapsed // SEGMENT_SECONDS, len(REGIONS) - 1)
        return REGIONS[index]

    def _update_message_for_region(self, progress: int) -> None:
        elapsed = TOTAL_SECONDS - self.remaining_seconds
        region_index = min(elapsed // SEGMENT_SECONDS, len(REGIONS) - 1)
        if self.remaining_seconds <= 0:
            self.title_var.set("你真棒，刷牙完成啦！")
        elif self.is_running and region_index != self.last_region_index:
            self.last_region_index = region_index
            if elapsed == 0:
                self.title_var.set(RUNNING_MESSAGES[0])
            elif progress >= 75:
                self.title_var.set(RUNNING_MESSAGES[2])
            else:
                self.title_var.set(RUNNING_MESSAGES[1])
        elif not self.is_running and self.remaining_seconds == TOTAL_SECONDS:
            self.title_var.set("准备好了吗？开始刷牙吧！")

    def _draw_progress_bar(self, progress: int) -> None:
        self.progress_canvas.delete("all")
        width = max(self.progress_canvas.winfo_width(), 600)
        bar_x1, bar_y1, bar_x2, bar_y2 = 10, 8, width - 10, 30
        self.progress_canvas.create_rectangle(bar_x1, bar_y1, bar_x2, bar_y2, fill="#F8E8EE", outline="")
        fill_width = bar_x1 + ((bar_x2 - bar_x1) * progress / 100)
        self.progress_canvas.create_rectangle(bar_x1, bar_y1, fill_width, bar_y2, fill="#FF7A59", outline="")

    def _complete_session(self) -> None:
        self.is_running = False
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.remaining_seconds = 0
        self.status_var.set("当前状态：已完成")
        self.title_var.set("你真棒，刷牙完成啦！")
        self.star_var.set("⭐ ⭐ ⭐ ⭐ ⭐")
        self._refresh_display()

        self.completed_today = self.storage.add_completed_session()
        self.today_var.set(f"今天已经刷牙 {self.completed_today} 次")
        praise = random.choice(ENCOURAGEMENTS)
        messagebox.showinfo("刷牙完成", praise)
        self.start_button.config(text="开始")

    def on_close(self) -> None:
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = KidBrushApp(root)
    app.progress_canvas.bind("<Configure>", lambda _event: app._refresh_display())
    root.mainloop()


if __name__ == "__main__":
    main()
