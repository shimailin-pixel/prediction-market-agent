"""pytest 根配置：确保项目根在 sys.path，使 `import src...` 可用。"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
