import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from floorplan_parser import FloorplanParser

def visualize_results(image_path):
    """Визуализация оригинала и результата"""
    
    parser = FloorplanParser()
    
    image = cv2.imread(str(image_path))
    result = parser.process(image_path)
    
    if not result:
        print("Не удалось получить результат для визуализации")
        return
    
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_img = original.copy()
    
    # ЦВЕТА:
    WALL_COLOR = (255, 0, 0)      # Красный для стен (в RGB)
    DOOR_COLOR = (0, 255, 0)    # Зеленый для дверей (в RGB)
    
    # Рисуем стены
    for wall in result.get('walls', []):
        pts = wall.get('points', [])
        if len(pts) >= 2:
            cv2.line(result_img, tuple(pts[0]), tuple(pts[1]), WALL_COLOR, 2)
    
    # Рисуем двери
    for door in result.get('doors', []):
        pts = door.get('points', [])
        if len(pts) >= 4:
            pts_np = np.array(pts, dtype=np.int32)
            # Рисуем заполненный прямоугольник
            cv2.fillPoly(result_img, [pts_np], DOOR_COLOR)
            # И контур
            cv2.polylines(result_img, [pts_np], True, (0, 0, 0), 2)
    
    # Отображение
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    ax1.imshow(original)
    ax1.set_title('Оригинал')
    ax1.axis('off')
    
    ax2.imshow(result_img)
    wall_count = len(result.get('walls', []))
    door_count = len(result.get('doors', []))
    ax2.set_title(f'Результат: {wall_count} стен, {door_count} дверей')
    ax2.axis('off')
    
    # Легенда
    legend_text = (
        f"Красный: стены\n"
        f"Зеленый: двери"
    )
    
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    if os.path.exists("plans"):
        images = list(Path("plans").glob("*.[pj][np]g"))
        if images:
            visualize_results(images[0])