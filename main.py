from floorplan_parser import FloorplanParser
import os
from pathlib import Path
import json

def main():
    """Основная функция"""
    parser = FloorplanParser()
    
    input_dir = "plans"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим изображения
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(Path(input_dir).glob(ext))
    
    images = images[:5]  
    
    if not images:
        print(f"В папке '{input_dir}' не найдены изображения")
        return
    
    print(f"Найдено изображений: {len(images)}")
    print("=" * 50)
    
    for img_path in images:
        print(f"Обработка: {img_path.name}")
        
        result = parser.process(img_path)
        if result:
            output_path = Path(output_dir) / f"{img_path.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Результат сохранен: {output_path}")
            print(f"Статистика:")
            print(f"  - Всего стен: {len(result['walls'])}")
            
            # Статистика по типам стен
            types = {}
            for wall in result['walls']:
                t = wall.get('type', 'other')
                types[t] = types.get(t, 0) + 1
            
            for t, count in types.items():
                print(f"       • {t}: {count}")
            
            print(f"     - Дверей: {len(result.get('doors', []))}")
        
        print("-" * 30)
    
    print("Обработка завершена!")