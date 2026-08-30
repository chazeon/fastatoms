# SPDX-License-Identifier: MIT
"""Keep the test run from writing anything into the source tree.

Importing a test module normally leaves a ``__pycache__/`` directory beside it.
This project keeps everything generated under ``build/``, so bytecode writing is
turned off here - before pytest imports any test module, which is what makes it
take effect for all of them.

Setting it in this file rather than relying on ``PYTHONDONTWRITEBYTECODE`` means
it holds however the suite is invoked: through ctest, through ``pytest`` run
from the repository root, or through an editor.
"""

import sys

sys.dont_write_bytecode = True
