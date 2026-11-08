import pyautogui
import time

print("Waiting for 5 seconds... Please focus on the window")
time.sleep(5)

current_position = pyautogui.position()
print(f"Current mouse position: {current_position}")

pyautogui.moveTo(1267, 454)
print("Moving the mouse to (1267, 454)")

time.sleep(2)

new_position = pyautogui.position()
print(f"New mouse position after moving: {new_position}")

current_position_after_move = pyautogui.position()
print(f"Current mouse position after move: {current_position_after_move}")
