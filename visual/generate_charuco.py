"""Generate ChArUco calibration board image for printing.

Board specs: 7x5, square=40mm, marker=30mm, DICT_5X5_50
Output: visual/charuco_board.png  (A4 landscape, 300 DPI — print at 100% scale)
"""

import pathlib
import numpy as np
import cv2
import cv2.aruco as aruco

_ROOT = pathlib.Path(__file__).parent.parent
OUTPUT = _ROOT / "visual" / "charuco_board.png"

# Board parameters (must match collect_calib_poses.py and solve_handeye.py)
COLS        = 7
ROWS        = 5
SQUARE_M    = 0.040   # 40 mm
MARKER_M    = 0.030   # 30 mm
ARUCO_DICT  = aruco.DICT_5X5_50

DPI         = 300
MM_PER_INCH = 25.4

def mm_to_px(mm):
    return int(round(mm * DPI / MM_PER_INCH))

def main():
    dictionary = aruco.Dictionary_get(ARUCO_DICT)
    board = aruco.CharucoBoard_create(COLS, ROWS, SQUARE_M, MARKER_M, dictionary)

    # Board pixel size at 300 DPI
    board_w_px = mm_to_px(COLS * SQUARE_M * 1000)   # 7*40 = 280 mm
    board_h_px = mm_to_px(ROWS * SQUARE_M * 1000)   # 5*40 = 200 mm

    # A4 landscape canvas: 297 x 210 mm
    canvas_w_px = mm_to_px(297)
    canvas_h_px = mm_to_px(210)

    board_img = board.draw((board_w_px, board_h_px))

    canvas = np.ones((canvas_h_px, canvas_w_px), dtype=np.uint8) * 255
    ox = (canvas_w_px - board_w_px) // 2
    oy = (canvas_h_px - board_h_px) // 2
    canvas[oy:oy + board_h_px, ox:ox + board_w_px] = board_img

    # Annotation
    text = "ChArUco 7x5 | square=40mm | marker=30mm | DICT_5X5_50 | Print at 100% scale (no fit-to-page)"
    cv2.putText(canvas, text, (ox, oy + board_h_px + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

    cv2.imwrite(str(OUTPUT), canvas)
    print(f"Saved: {OUTPUT}")
    print(f"Board size: {COLS*40} x {ROWS*40} mm  ({board_w_px} x {board_h_px} px)")
    print("IMPORTANT: Print at EXACTLY 100% scale (disable 'fit to page')")

if __name__ == "__main__":
    main()
