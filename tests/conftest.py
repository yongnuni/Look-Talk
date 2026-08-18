"""로컬 Tcl/Tk가 없는 테스트 환경의 화면 크기 전용 대체물."""

import tkinter as tk


try:
    _probe_root = tk.Tk()
except tk.TclError:
    class _HeadlessTestRoot:
        def withdraw(self):
            pass

        def winfo_screenwidth(self):
            return 1920

        def winfo_screenheight(self):
            return 1080

        def destroy(self):
            pass

    tk.Tk = _HeadlessTestRoot
else:
    _probe_root.withdraw()
    _probe_root.destroy()
