"""
Application GUI de Reconnaissance de Caractères Manuscrits (ICR)
Utilise PyQt5 pour l'interface graphique et un modèle CNN PyTorch pour la prédiction
"""

import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QGroupBox, QGridLayout,
    QProgressBar, QMessageBox, QSplitter, QStatusBar, QAction, QMenuBar
)
from PyQt5.QtCore import Qt, QSize, QPoint, QRect
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QFont, QIcon, QPalette
)


# ============================================================================
# ARCHITECTURE DU MODÈLE CNN (identique à l'entraînement)
# ============================================================================

class AdvancedCNN(nn.Module):
    """
    CNN profond pour reconnaissance de caractères manuscrits.
    Architecture : 3 blocs convolutionnels + Fully Connected + Dropout
    """
    def __init__(self, num_classes=26):
        super(AdvancedCNN, self).__init__()

        # Bloc Convolutionnel 1 : 1 -> 32 filtres
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc Convolutionnel 2 : 32 -> 64 filtres
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc Convolutionnel 3 : 64 -> 128 filtres
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dimension après convolutions : 128 × 3 × 3 = 1152
        self.fc_input_size = 128 * 3 * 3

        # Couches Fully Connected
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# ============================================================================
# WIDGET DE DESSIN (Canvas)
# ============================================================================

class DrawingCanvas(QLabel):
    """
    Widget pour dessiner des caractères manuscrits à la souris.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 3px solid #3498db;
                border-radius: 10px;
            }
        """)
        
        # Canvas de dessin
        self.canvas = QPixmap(280, 280)
        self.canvas.fill(Qt.white)
        self.setPixmap(self.canvas)
        
        # Paramètres de dessin
        self.drawing = False
        self.last_point = QPoint()
        self.pen_size = 18
        self.pen_color = Qt.black
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() & Qt.LeftButton:
            painter = QPainter(self.canvas)
            pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.last_point = event.pos()
            self.setPixmap(self.canvas)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            
    def clear_canvas(self):
        """Efface le canvas."""
        self.canvas.fill(Qt.white)
        self.setPixmap(self.canvas)
        
    def get_image(self):
        """
        Retourne l'image du canvas comme array NumPy prétraité pour le modèle.
        """
        # Convertir QPixmap en QImage
        qimage = self.canvas.toImage()
        
        # Convertir en format compatible
        qimage = qimage.convertToFormat(QImage.Format_Grayscale8)
        
        # Obtenir les dimensions
        width = qimage.width()
        height = qimage.height()
        
        # Convertir en bytes puis en NumPy array
        ptr = qimage.bits()
        ptr.setsize(width * height)
        arr = np.array(ptr).reshape(height, width)
        
        # Redimensionner en 28x28
        img_pil = Image.fromarray(arr)
        img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Inverser (fond blanc -> fond noir pour correspondre au format d'entraînement)
        img_array = 1.0 - img_array
        
        return img_array


# ============================================================================
# WIDGET D'AFFICHAGE DES RÉSULTATS
# ============================================================================

class ResultWidget(QFrame):
    """
    Widget pour afficher les résultats de prédiction.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Label pour la lettre prédite
        self.letter_label = QLabel("?")
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.letter_label.setFont(QFont("Arial", 72, QFont.Bold))
        self.letter_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(self.letter_label)
        
        # Label pour la confiance
        self.confidence_label = QLabel("Confiance: ---%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 14))
        self.confidence_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.confidence_label)
        
        # Barre de progression pour la confiance
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background-color: #ecf0f1;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.confidence_bar)
        
    def update_result(self, letter, confidence):
        """Met à jour l'affichage avec les résultats."""
        self.letter_label.setText(letter)
        confidence_pct = confidence * 100
        self.confidence_label.setText(f"Confiance: {confidence_pct:.1f}%")
        self.confidence_bar.setValue(int(confidence_pct))
        
        # Changer la couleur selon la confiance
        if confidence_pct >= 80:
            color = "#27ae60"  # Vert
        elif confidence_pct >= 50:
            color = "#f39c12"  # Orange
        else:
            color = "#e74c3c"  # Rouge
            
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background-color: #ecf0f1;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        
    def clear(self):
        """Réinitialise l'affichage."""
        self.letter_label.setText("?")
        self.confidence_label.setText("Confiance: ---%")
        self.confidence_bar.setValue(0)


# ============================================================================
# WIDGET TOP-5 PRÉDICTIONS
# ============================================================================

class Top5Widget(QGroupBox):
    """
    Widget pour afficher les 5 meilleures prédictions.
    """
    def __init__(self, parent=None):
        super().__init__("Top 5 Prédictions", parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #2980b9;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Labels pour les 5 prédictions
        self.prediction_bars = []
        self.prediction_labels = []
        
        for i in range(5):
            row_layout = QHBoxLayout()
            
            # Rang + Lettre
            label = QLabel(f"{i+1}. ?")
            label.setFont(QFont("Arial", 12))
            label.setFixedWidth(60)
            row_layout.addWidget(label)
            self.prediction_labels.append(label)
            
            # Barre de progression
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v%")
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    background-color: #ecf0f1;
                    height: 18px;
                }
                QProgressBar::chunk {
                    background-color: #3498db;
                    border-radius: 2px;
                }
            """)
            row_layout.addWidget(bar)
            self.prediction_bars.append(bar)
            
            layout.addLayout(row_layout)
            
    def update_predictions(self, predictions):
        """
        Met à jour les prédictions.
        predictions: liste de tuples (lettre, probabilité)
        """
        for i, (letter, prob) in enumerate(predictions[:5]):
            self.prediction_labels[i].setText(f"{i+1}. {letter}")
            self.prediction_bars[i].setValue(int(prob * 100))
            
            # Colorer la première prédiction différemment
            if i == 0:
                self.prediction_bars[i].setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #bdc3c7;
                        border-radius: 3px;
                        background-color: #ecf0f1;
                        height: 18px;
                    }
                    QProgressBar::chunk {
                        background-color: #27ae60;
                        border-radius: 2px;
                    }
                """)
            else:
                self.prediction_bars[i].setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #bdc3c7;
                        border-radius: 3px;
                        background-color: #ecf0f1;
                        height: 18px;
                    }
                    QProgressBar::chunk {
                        background-color: #3498db;
                        border-radius: 2px;
                    }
                """)
                
    def clear(self):
        """Réinitialise l'affichage."""
        for i in range(5):
            self.prediction_labels[i].setText(f"{i+1}. ?")
            self.prediction_bars[i].setValue(0)


# ============================================================================
# FENÊTRE PRINCIPALE
# ============================================================================

class ICRMainWindow(QMainWindow):
    """
    Fenêtre principale de l'application ICR.
    """
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ICR - V1 - Reconnaissance de Caractères Manuscrits")
        self.setMinimumSize(900, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
        """)
        
        # Initialisation du modèle
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Interface
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        
        # Charger le modèle
        self.load_model()
        
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ===== PANNEAU GAUCHE (Dessin) =====
        left_panel = QGroupBox("Zone de Dessin")
        left_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #9b59b6;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #8e44ad;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        
        # Instructions
        instructions = QLabel("Dessinez une lettre (A-Z) avec la souris")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setFont(QFont("Arial", 11))
        instructions.setStyleSheet("color: #7f8c8d; margin-bottom: 10px;")
        left_layout.addWidget(instructions)
        
        # Canvas de dessin
        self.canvas = DrawingCanvas()
        canvas_container = QHBoxLayout()
        canvas_container.addStretch()
        canvas_container.addWidget(self.canvas)
        canvas_container.addStretch()
        left_layout.addLayout(canvas_container)
        
        # Boutons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.predict_btn = QPushButton("🔍 Prédire")
        self.predict_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_btn.setFixedHeight(45)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.predict_btn.clicked.connect(self.predict_drawing)
        buttons_layout.addWidget(self.predict_btn)
        
        self.clear_btn = QPushButton("🗑️ Effacer")
        self.clear_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.clear_btn.setFixedHeight(45)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_all)
        buttons_layout.addWidget(self.clear_btn)
        
        left_layout.addLayout(buttons_layout)
        
        # Bouton charger image
        self.load_image_btn = QPushButton("📁 Charger une Image")
        self.load_image_btn.setFont(QFont("Arial", 11))
        self.load_image_btn.setFixedHeight(40)
        self.load_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #5dade2;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """)
        self.load_image_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_image_btn)
        
        main_layout.addWidget(left_panel)
        
        # ===== PANNEAU DROIT (Résultats) =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Résultat principal
        result_group = QGroupBox("Résultat de la Prédiction")
        result_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #27ae60;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #1e8449;
            }
        """)
        result_layout = QVBoxLayout(result_group)
        
        self.result_widget = ResultWidget()
        result_layout.addWidget(self.result_widget)
        
        right_layout.addWidget(result_group)
        
        # Top 5 prédictions
        self.top5_widget = Top5Widget()
        right_layout.addWidget(self.top5_widget)
        
        # Aperçu de l'image prétraitée
        preview_group = QGroupBox("Image Prétraitée (28×28)")
        preview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #95a5a6;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #7f8c8d;
            }
        """)
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(84, 84)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #34495e;
                border-radius: 5px;
            }
        """)
        preview_container = QHBoxLayout()
        preview_container.addStretch()
        preview_container.addWidget(self.preview_label)
        preview_container.addStretch()
        preview_layout.addLayout(preview_container)
        
        right_layout.addWidget(preview_group)
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel)
        
    def setup_menu(self):
        """Configure la barre de menu."""
        menubar = self.menuBar()
        
        # Menu Fichier
        file_menu = menubar.addMenu("Fichier")
        
        load_model_action = QAction("Charger un modèle...", self)
        load_model_action.triggered.connect(self.browse_model)
        file_menu.addAction(load_model_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quitter", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Menu Aide
        help_menu = menubar.addMenu("Aide")
        
        about_action = QAction("À propos", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_statusbar(self):
        """Configure la barre de statut."""
        self.statusBar().showMessage("Prêt")
        
        # Label pour l'état du modèle
        self.model_status_label = QLabel()
        self.statusBar().addPermanentWidget(self.model_status_label)
        
        # Label pour le device
        device_label = QLabel(f"Device: {self.device}")
        device_label.setStyleSheet("color: #7f8c8d; margin-right: 10px;")
        self.statusBar().addPermanentWidget(device_label)
        
    def load_model(self, model_path=None):
        """Charge le modèle CNN."""
        if model_path is None:
            # Chercher le modèle dans le répertoire courant ou parent
            possible_paths = [
                'icr_cnn_model.pth',
                '../icr_cnn_model.pth',
                os.path.join(os.path.dirname(__file__), 'icr_cnn_model.pth'),
                os.path.join(os.path.dirname(__file__), '..', 'icr_cnn_model.pth'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = AdvancedCNN(num_classes=26).to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.model_status_label.setText(f"✅ Modèle chargé: {os.path.basename(model_path)}")
                self.model_status_label.setStyleSheet("color: #27ae60;")
                self.statusBar().showMessage(f"Modèle chargé avec succès depuis {model_path}", 3000)
                
            except Exception as e:
                self.model = None
                self.model_status_label.setText("❌ Erreur de chargement")
                self.model_status_label.setStyleSheet("color: #e74c3c;")
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le modèle:\n{str(e)}")
        else:
            self.model = None
            self.model_status_label.setText("⚠️ Aucun modèle chargé")
            self.model_status_label.setStyleSheet("color: #f39c12;")
            self.statusBar().showMessage("Aucun modèle trouvé. Utilisez Fichier > Charger un modèle...")
            
    def browse_model(self):
        """Ouvre un dialogue pour sélectionner un fichier modèle."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner le fichier modèle",
            "",
            "Fichiers PyTorch (*.pth *.pt);;Tous les fichiers (*)"
        )
        if file_path:
            self.load_model(file_path)
            
    def predict_drawing(self):
        """Effectue une prédiction sur le dessin actuel."""
        if self.model is None:
            QMessageBox.warning(self, "Attention", "Aucun modèle n'est chargé!\nVeuillez charger un modèle via Fichier > Charger un modèle...")
            return
            
        try:
            # Obtenir l'image prétraitée
            img_array = self.canvas.get_image()
            
            # Afficher l'aperçu
            self.show_preview(img_array)
            
            # Conversion en tenseur
            img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Prédiction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Top-5 prédictions
                top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
                top5_probs = top5_probs.cpu().numpy()[0]
                top5_indices = top5_indices.cpu().numpy()[0]
                
                # Meilleure prédiction
                predicted_idx = top5_indices[0]
                confidence = top5_probs[0]
                predicted_letter = chr(65 + predicted_idx)
                
            # Mettre à jour l'interface
            self.result_widget.update_result(predicted_letter, confidence)
            
            top5_results = [(chr(65 + idx), prob) for idx, prob in zip(top5_indices, top5_probs)]
            self.top5_widget.update_predictions(top5_results)
            
            self.statusBar().showMessage(f"Prédiction: {predicted_letter} ({confidence*100:.1f}%)", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la prédiction:\n{str(e)}")
            
    def show_preview(self, img_array):
        """Affiche l'aperçu de l'image prétraitée."""
        # Convertir en 0-255
        img_display = (img_array * 255).astype(np.uint8)
        
        # Créer QImage
        height, width = img_display.shape
        qimage = QImage(img_display.data, width, height, width, QImage.Format_Grayscale8)
        
        # Redimensionner pour l'affichage
        pixmap = QPixmap.fromImage(qimage).scaled(84, 84, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)
        
    def load_image(self):
        """Charge une image depuis un fichier."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;Tous les fichiers (*)"
        )
        
        if file_path:
            try:
                # Charger l'image
                img = Image.open(file_path).convert('L')
                img_resized = img.resize((280, 280), Image.Resampling.LANCZOS)
                
                # Convertir en QPixmap pour affichage
                img_array = np.array(img_resized)
                qimage = QImage(img_array.data, 280, 280, 280, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
                
                # Mettre à jour le canvas
                self.canvas.canvas = pixmap
                self.canvas.setPixmap(pixmap)
                
                self.statusBar().showMessage(f"Image chargée: {os.path.basename(file_path)}", 3000)
                
                # Prédire automatiquement
                self.predict_drawing()
                
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de charger l'image:\n{str(e)}")
                
    def clear_all(self):
        """Efface le canvas et les résultats."""
        self.canvas.clear_canvas()
        self.result_widget.clear()
        self.top5_widget.clear()
        self.preview_label.clear()
        self.statusBar().showMessage("Effacé", 2000)
        
    def show_about(self):
        """Affiche la boîte de dialogue À propos."""
        QMessageBox.about(
            self,
            "À propos de ICR",
            """<h2>ICR - Reconnaissance de Caractères Manuscrits</h2>
            <p><b>Version:</b> 1.0</p>
            <p>Application de reconnaissance de caractères manuscrits (A-Z) 
            utilisant un réseau de neurones convolutif (CNN) entraîné sur 
            les datasets A-Z Handwritten Alphabets et EMNIST.</p>
            <p><b>Fonctionnalités:</b></p>
            <ul>
                <li>Dessin de caractères à la souris</li>
                <li>Chargement d'images externes</li>
                <li>Prédiction en temps réel</li>
                <li>Affichage des Top-5 prédictions</li>
            </ul>
            <p><b>Technologies:</b> PyQt5, PyTorch, NumPy, PIL</p>
            """
        )


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Point d'entrée de l'application."""
    app = QApplication(sys.argv)
    
    # Style global
    app.setStyle('Fusion')
    
    # Palette de couleurs
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 246, 250))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    app.setPalette(palette)
    
    # Créer et afficher la fenêtre principale
    window = ICRMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
