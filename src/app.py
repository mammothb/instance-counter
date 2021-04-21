import sys

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QWidget,
)


class LayoutDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 300
        self.top = 300
        self.width = 400
        self.height = 450

        # self.work = WorkThread()
        self.init_widgets().init_layout().init_event()

    def init_widgets(self):
        self.setWindowTitle(self.tr("Instance Counter"))
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label_filename = QLabel(self.tr("Video path:"))
        self.line_edit_filename = QLineEdit()
        self.button_browse = QPushButton(self.tr("Browse"))

        # self.movie_source_label = QLabel(self.tr("选择片源:"))
        # self.movie_source_combobox = QComboBox()
        # self.movie_source_combobox.addItem(self.tr('电影天堂'))

        self.button_process = QPushButton(self.tr("Process"))

        # self.tip_label = QLabel(self.tr("未开始查询..."))
        # self.search_content_label = QLabel(self.tr("查询内容:"))
        # self.search_content_text_list = QListWidget()

        # self.menu_bar = self.menuBar()

        return self

    def init_layout(self):
        layout = QGridLayout()
        layout.addWidget(self.label_filename, 0, 0)
        layout.addWidget(self.line_edit_filename, 0, 1)
        layout.addWidget(self.button_browse, 0, 2)
        layout.addWidget(self.button_process, 0, 3)

        frame = QWidget()
        self.setCentralWidget(frame)
        frame.setLayout(layout)
        return self

    def init_event(self):
        self.button_browse.clicked.connect(self.browse_file)

    def browse_file(self):
        filename = QFileDialog.getOpenFileName(
            self, "Open file", "C:\\", "Video files (*.mp4)"
        )
        self.line_edit_filename.setText(filename[0])

app = QApplication(sys.argv)
dialog = LayoutDialog()
dialog.show()
app.exec_() 