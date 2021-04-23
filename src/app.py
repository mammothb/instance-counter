import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStyle,
    QWidget,
)


class LayoutDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 300
        self.top = 300
        self.width = 640
        self.height = 480

        # self.work = WorkThread()
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
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
        self.video_widget = QVideoWidget()
        self.button_play = QPushButton()
        self.button_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.button_play.setEnabled(False)

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

        layout.addWidget(self.video_widget, 1, 0, 2, 4)
        layout.addWidget(self.button_play, 3, 0)

        frame = QWidget()
        self.setCentralWidget(frame)
        frame.setLayout(layout)

        self.media_player.setVideoOutput(self.video_widget)

        return self

    def init_event(self):
        self.button_browse.clicked.connect(self.browse_file)
        self.button_play.clicked.connect(self.play)

    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open file", "C:\\", "Video files (*.mp4 *.avi)"
        )
        if filename != "":
            self.line_edit_filename.setText(filename)
            self.media_player.setMedia(
                QMediaContent(QUrl.fromLocalFile(filename))
            )
            self.button_play.setEnabled(True)

    def play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

app = QApplication(sys.argv)
dialog = LayoutDialog()
dialog.show()
app.exec_() 