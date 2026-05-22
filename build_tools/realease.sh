pyinstaller --onefile --noconsole --add-data "src/icon/;src/icon/" --add-data "style/;style/" main.py
pyinstaller main.spec
pyside6-deploy -c pysidedeploy.spec --windows-disable-console
nuitka --standalone --show-memory --show-progress --nofollow-imports --follow-import-to=netron --output-dir=build --windows-disable-console --enable-plugin=pyside6 main.py
nuitka --standalone --show-memory --show-progress --nofollow-imports --follow-import-to=netron --output-dir=build --enable-plugin=pyside6 main.py
