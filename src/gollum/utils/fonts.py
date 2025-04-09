from pathlib import Path
from matplotlib import font_manager

def register_font(relative_font_path="fonts/NotoSansCoptic-Regular.ttf"):
    path = Path(__file__).resolve() if '__file__' in globals() else Path().resolve()

    while not (path / "fonts").exists() and path != path.parent:
        path = path.parent

    font_path = path / relative_font_path
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found at {font_path}")
    
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    print(f"✅ Registered font: {font_name} from {font_path}")
    
    return font_name