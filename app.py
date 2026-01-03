import pyautogui
import time

# Wait for 5 seconds to give you time to focus on the window
print("Waiting for 5 seconds... Please focus on the window")
time.sleep(5)

# Get and print the current mouse position
current_position = pyautogui.position()
print(f"Current mouse position: {current_position}")

# Move the mouse to the coordinates (1267, 454)
pyautogui.moveTo(1267, 454)
print("Moving the mouse to (1267, 454)")

# Wait to visually confirm the mouse movement
time.sleep(2)

# Print the new position after moving
new_position = pyautogui.position()
print(f"New mouse position after moving: {new_position}")

# Finally, print the current position again after the move
current_position_after_move = pyautogui.position()
print(f"Current mouse position after move: {current_position_after_move}")
