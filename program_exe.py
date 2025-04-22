from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QLineEdit, QPushButton, QListWidget, QMessageBox, QHBoxLayout, QSlider
)
from PyQt5.QtCore import Qt
import sys
import numpy as np
from pymcdm.methods import TOPSIS

criteria = ["재료 특성", "필요 정밀도", "운항 영향도", "공급가", "공급망 범위", "비체적"]
weights = np.array([0.25, 0.2, 0.25, 0.1, 0.1, 0.1])
cost_criteria = np.array([0, 0, 0, 1, 0, 1])

spare_parts = {}

class TopsisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOPSIS 부품 평가기")
        self.setGeometry(100, 100, 400, 500)

        self.layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("부품 이름 입력")
        self.layout.addWidget(self.name_input)

        self.sliders = []
        self.slider_values = []
        for crit in criteria:
            label = QLabel(crit)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setValue(5)
            self.sliders.append(slider)
            self.layout.addWidget(label)
            self.layout.addWidget(slider)

        self.add_button = QPushButton("부품 추가")
        self.add_button.clicked.connect(self.add_part)
        self.layout.addWidget(self.add_button)

        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        self.eval_button = QPushButton("TOPSIS 평가")
        self.eval_button.clicked.connect(self.evaluate)
        self.layout.addWidget(self.eval_button)

        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def add_part(self):
        name = self.name_input.text()
        if not name:
            QMessageBox.warning(self, "경고", "부품 이름을 입력하세요.")
            return
        values = [slider.value() for slider in self.sliders]
        spare_parts[name] = values
        self.list_widget.addItem(name)
        QMessageBox.information(self, "추가 완료", f"{name}이(가) 추가되었습니다.")

    def evaluate(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "선택 필요", "평가할 부품을 선택하세요.")
            return
        name = selected_items[0].text()
        user_part = spare_parts[name]

        # 대안
        alt = np.array([
            [8, 7, 6, 6, 7, 5],
            [7, 8, 8, 7, 8, 6],
            [7, 7, 9, 5, 9, 8]
        ])
        matrix = np.vstack([user_part, alt])
        topsis = TOPSIS()
        scores = topsis(matrix, weights, cost_criteria)

        options = ["(입력부품)", "제작", "위급시 제작", "선적"]
        result_text = "\n".join(f"{opt}: {score:.3f}" for opt, score in zip(options, scores))
        self.result_label.setText(result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TopsisApp()
    window.show()
    sys.exit(app.exec_())
