import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
import numpy as np
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
from torchvision.models import resnet101, densenet121

def get_model_dict():
    models = dict()

    resnet_u0 = resnet101(num_classes=14)
    resnet_u0.fc = torch.nn.Linear(in_features=resnet_u0.fc.in_features, out_features=14)
    resnet_u0.load_state_dict(torch.load('checkpoint/ResNetTL-U0.pth', weights_only=True))

    models['ResNetTL-U0'] = resnet_u0

    resnet_u1 = resnet101(num_classes=14)
    resnet_u1.fc = torch.nn.Linear(in_features=resnet_u1.fc.in_features, out_features=14)
    resnet_u1.load_state_dict(torch.load('checkpoint/ResNetTL-U1.pth', weights_only=True))

    models['ResNetTL-U1'] = resnet_u1

    densenet_u0 = densenet121(num_classes=14)
    densenet_u0.classifier = torch.nn.Linear(in_features=densenet_u0.classifier.in_features, out_features=14)
    densenet_u0.load_state_dict(torch.load('checkpoint/DenseNetTL-U0.pth', weights_only=True))

    models['DenseNetTL-U0'] = densenet_u0

    densenet_u1 = densenet121(num_classes=14)
    densenet_u1.classifier = torch.nn.Linear(in_features=densenet_u1.classifier.in_features, out_features=14)
    densenet_u1.load_state_dict(torch.load('checkpoint/DenseNetTL-U1.pth', weights_only=True))

    models['DenseNetTL-U1'] = densenet_u1

    return models

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chest X-ray Diseases Classification")
        self.setGeometry(100, 100, 1200, 800)

        # main layout
        main_layout = QHBoxLayout()

        # main display area
        central_widget = QVBoxLayout()

        # model selection dropdown
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setFont(QFont('Constantia', 16))
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(["None", "DenseNetTL-U0", "DenseNetTL-U1", "ResNetTL-U0", "ResNetTL-U1"])
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.model_combo_box)
        central_widget.addLayout(model_selection_layout)

        # display label
        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap().scaled(800, 800, Qt.KeepAspectRatio))
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f5f5f5;")
        central_widget.addWidget(self.image_label, 5)

        # upload and predict
        button_layout = QHBoxLayout()
        upload_button = QPushButton("Upload")
        upload_button.setFont(QFont('Constantia', 20))
        upload_button.setStyleSheet("background-color: #6AB187; color: black; font-size: 25px; border-radius: 5px;")
        upload_button.clicked.connect(self.upload_image)  # Connect to upload function
        report_button = QPushButton("Predict")
        report_button.setFont(QFont('Constantia', 20))
        report_button.setStyleSheet("background-color: #4CB5F5; color: black; font-size: 25px; border-radius: 5px;")
        report_button.clicked.connect(self.predict)  # Connect to AI model
        button_layout.addWidget(upload_button)
        button_layout.addWidget(report_button)
        central_widget.addLayout(button_layout)

        main_layout.addLayout(central_widget, 4)

        # output panel
        self.output_panel = QVBoxLayout()
        output_label = QLabel("OUTPUT")
        output_label.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #333; color: white; padding: 5px;")
        self.output_panel.addWidget(output_label)

        label1 = QLabel("Disease Indicators: ")
        label1.setFont(QFont('Constantia', 24))
        label1.setStyleSheet("font-size: 18px; color: #333;")
        
        # output results
        output_layout = QVBoxLayout()
        output_layout.addWidget(label1)
        self.progress_bars = {}
        self.conditions = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly', 
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation', 
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices'
        ]
        for condition in self.conditions:
            row_layout = QHBoxLayout()
            label = QLabel(f"{condition}:")
            label.setFont(QFont('Constantia', 16))
            label.setStyleSheet("font-size: 14px; color: #555;")
            label.setFixedWidth(200)

            progress = QProgressBar()
            progress.setValue(0)
            progress.setFormat("%v")
            progress.setFixedWidth(300)
            progress.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #bbb;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #76c7c0;
                    width: 20px;
                }
            """)
            progress.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        

            row_layout.addWidget(label)
            row_layout.addWidget(progress)

            output_layout.addLayout(row_layout)
            self.progress_bars[condition] = progress
        # update predicted and true labels
        label2 = QLabel("Predicted Diseases: ")
        label2.setFont(QFont('Constantia', 24))
        label2.setStyleSheet("font-size: 18px; color: #333;")

        label3 = QLabel("True Diseases: ")
        label3.setFont(QFont('Constantia', 24))
        label3.setStyleSheet("font-size: 18px; color: #333;")

        self.predicted_text = QTextEdit()
        self.predicted_text.setReadOnly(True)
        self.predicted_text.setStyleSheet("font-size: 16px; color: #555;")
        
        self.true_text = QTextEdit()
        self.true_text.setReadOnly(True)
        self.true_text.setStyleSheet("font-size: 16px; color: #555;")
        # add the labels to the layout after the progress bars
        
        output_layout.addWidget(label2)
        output_layout.addWidget(self.predicted_text)
        output_layout.addWidget(label3)
        output_layout.addWidget(self.true_text)

        main_layout.addLayout(output_layout)
        
        # set the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.image_path = None  # store the uploaded image path
        self.model_dict = get_model_dict()

    def upload_image(self):
        """Open a file dialog to upload an image and display it."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Image", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image = image.convert("RGBA")
            
            # Convert Pillow image to QImage
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
            
            # Convert QImage to QPixmap for display
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(800, 800, Qt.KeepAspectRatio))

    def predict(self):
        """Predict image labels and update the output panel."""
        if not self.image_path:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("Please upload an image first.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        # load image for  model
        img_path = self.image_path
        image = Image.open(img_path).convert('RGB')
        image = transforms(image=np.array(image))['image']

        # select a model
        selected_model = self.model_combo_box.currentText()
        if selected_model == 'None':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("Please select a model first.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        # get true labels
        try:
            labels = pd.read_csv(f'test/u{selected_model[-1]}/u{selected_model[-1]}_test.csv', index_col=0)
        
        except FileNotFoundError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("The image is not from the test set.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        image_index = 'chexpert' + self.image_path.split('chexpert', 1)[-1]
        try:
            image_label = labels.loc[image_index]
        except KeyError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("The image is not from the test set.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        positive_labels = image_label[image_label == 1]

        # get thresholds
        thresholds = pd.read_csv(f'checkpoint/{selected_model}.csv')['threshold']

        # load model
        model = self.model_dict[selected_model]
        model.eval()
        
        # get predicted labels
        results = model(image.unsqueeze(0))
        results = nn.Sigmoid()(results)

        pred_postive_index = np.where(results.squeeze().detach().numpy() > thresholds)[0]

        for i, value in enumerate(results[0]):
            self.progress_bars[self.conditions[i]].setValue(round(value.item() * 100))

        self.predicted_text.setText(','.join(self.conditions[i] for i in pred_postive_index))
        self.true_text.setText(', '.join(positive_labels.index))

transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize([0.506, 0.506, 0.506], [0.287, 0.287, 0.287]),
        ToTensorV2()
    ])


# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
