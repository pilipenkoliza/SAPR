import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QGridLayout, QComboBox, QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QTextEdit
)
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt
import math
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt


class Rod:
    def __init__(self, start_node, end_node, E, A, L, q_p=0):
        self.start_node = start_node
        self.end_node = end_node
        self.E = E
        self.A = A
        self.L = L
        self.q_p = q_p

    def local_stiffness_matrix(self):
        k = self.E * self.A / self.L
        return np.array([[k, -k], [-k, k]])

    def local_load_vector(self):
        return np.array([-self.q_p * self.L / 2, -self.q_p * self.L / 2])


class Node:
    def __init__(self, x: int, y, force=0, fixed=False) -> None:
        self.x = x
        self.y = y
        self.force = force
        self.fixed: bool = fixed


class Processor:
    def __init__(self, rods, num_nodes, forces, displacements_bc=None):
        self.rods = rods
        self.num_nodes = num_nodes
        self.forces = forces
        self.displacements_bc = displacements_bc or {}

    def assemble_global_stiffness_matrix(self):
        K_global = lil_matrix((self.num_nodes, self.num_nodes))

        for rod in self.rods:
            k_local = rod.local_stiffness_matrix()
            indices = [rod.start_node, rod.end_node]
            for i in range(2):
                for j in range(2):
                    K_global[indices[i], indices[j]] += k_local[i, j]

        return K_global.tocsr()

    def assemble_global_load_vector(self):
        F_global = np.zeros(self.num_nodes)

        # Учитываем распределённую нагрузку для каждого стержня
        for rod in self.rods:
            q = rod.q_p  # распределённая нагрузка
            L = rod.L  # длина стержня
            f_local = np.array([q * L / 2, q * L / 2])  # распределённая нагрузка по узлам
            indices = [rod.start_node, rod.end_node]

            for i in range(2):
                F_global[indices[i]] += f_local[i]

        # Добавляем сосредоточенные силы
        for node, force in self.forces.items():
            F_global[node] += force

        return F_global

    def solve(self):
        # Сборка матрицы жесткости и вектора нагрузок
        K_global = self.assemble_global_stiffness_matrix().tolil()
        F_global = self.assemble_global_load_vector()

        # Применение граничных условий
        for node, displacement in self.displacements_bc.items():
            K_global[node, :] = 0
            K_global[node, node] = 1
            F_global[node] = displacement

        K_global = K_global.tocsr()  # Конвертируем обратно в csr_matrix для решения

        try:
            displacements = spsolve(K_global, F_global)
            return displacements
        except np.linalg.LinAlgError:
            print("Система уравнений не имеет единственного решения.")
            return None

    def calculate_stresses_and_forces(self, displacements):
        forces = []
        stresses = []

        for rod in self.rods:
            start_node = rod.start_node
            end_node = rod.end_node
            displacement_start = displacements[start_node]
            displacement_end = displacements[end_node]

            # Продольная сила с учётом распределённой нагрузки
            x = 0
            N_x = rod.E * rod.A / rod.L * (displacement_end - displacement_start) + (rod.q_p * rod.L / 2) * (1 - 2 * x / rod.L)
            N_x_end = N_x - rod.q_p * rod.L
            forces_n_x = [N_x, N_x_end]
            forces.append(forces_n_x)

            sigma_x = N_x / rod.A
            sigma_x_end = sigma_x - rod.q_p * rod.L / rod.A
            stresses_x = [sigma_x, sigma_x_end]
            stresses.append(stresses_x)

        return forces, stresses

    def get_results(self):
        displacements = self.solve()
        if displacements is None:
            return None, [], []

        forces, stresses = self.calculate_stresses_and_forces(displacements)

        output = "\n".join(
            f"E: {rod.E}, A: {rod.A}, dis_end: {displacements[rod.end_node]}, "
            f"dis_start: {displacements[rod.start_node]}, q: {rod.q_p}, L: {rod.L}"
            for rod in self.rods
        )

        # Печатаем или возвращаем вывод
        print(output)

        # Возвращаем перемещения, силы и напряжения
        return displacements, forces, stresses


class Preprocessor(QWidget):
    def __init__(self):
        super().__init__()
        self.rods = []
        self.nodes = []
        self.scale_factor = 1.0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Препроцессор для плоских стержневых конструкций")

        self.rod_grid = QGridLayout()
        self.node_grid = QGridLayout()

        self.add_rod_button = QPushButton("Добавить стержень")
        self.add_rod_button.clicked.connect(self.add_rod_row)
        self.add_node_button = QPushButton("Добавить узел")
        self.add_node_button.clicked.connect(self.add_node_row)
        self.run_button = QPushButton("Рассчитать")
        self.run_button.clicked.connect(self.run_processor)
        self.save_button = QPushButton("Сохранить проект")
        self.save_button.clicked.connect(self.create_project_file)

        self.load_button = QPushButton("Загрузить проект")
        self.load_button.clicked.connect(self.load_project_file)

        self.canvas = QLabel()
        self.canvas.setMinimumSize(500, 500)
        self.canvas.setStyleSheet("background-color: white;")
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Данные о стержнях:"))
        main_layout.addLayout(self.rod_grid)
        main_layout.addWidget(self.add_rod_button)
        main_layout.addWidget(QLabel("Данные об узлах:"))
        main_layout.addLayout(self.node_grid)
        main_layout.addWidget(self.add_node_button)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.save_button)
        main_layout.addWidget(self.run_button)
        main_layout.addWidget(self.load_button)

        self.setLayout(main_layout)

    def check_rod(self, start, end):
        for rod in self.rods:
            if (rod.start_node == start and rod.end_node == end) or (rod.end_node == start and rod.start_node == end):
                return True

        return False

    def add_rod_row(self):
        row = self.rod_grid.rowCount()
        area_input = QLineEdit()
        material_input = QLineEdit()
        distributed_load_input = QLineEdit()
        start_node_input = QLineEdit()
        end_node_input = QLineEdit()

        self.rod_grid.addWidget(QLabel(f"Стержень "), row, 0)
        self.rod_grid.addWidget(QLabel(f"{row}:"), row, 1)
        self.rod_grid.addWidget(QLabel("Площадь:"), row, 2)
        self.rod_grid.addWidget(area_input, row, 3)
        self.rod_grid.addWidget(QLabel("Модуль упругости:"), row, 4)
        self.rod_grid.addWidget(material_input, row, 5)
        self.rod_grid.addWidget(QLabel("Расп. нагрузка:"), row, 6)
        self.rod_grid.addWidget(distributed_load_input, row, 7)
        self.rod_grid.addWidget(QLabel("Нач. узел:"), row, 8)
        self.rod_grid.addWidget(start_node_input, row, 9)
        self.rod_grid.addWidget(QLabel("Кон. узел:"), row, 10)
        self.rod_grid.addWidget(end_node_input, row, 11)

        # Ищем стержень по его индексу и обновляем его данные
        def update_rod_data():
            try:
                area = float(area_input.text())
                material = float(material_input.text())
                distributed_load = float(distributed_load_input.text())
                start_node = int(start_node_input.text()) - 1
                end_node = int(end_node_input.text()) - 1
                rod_index = int(self.node_grid.itemAtPosition(row, 1).widget().text()[:-1]) - 1

                # Проверка на положительные значения для площади и модуля упругости
                if area <= 0:
                    raise ValueError("Площадь должна быть больше нуля.")
                if material <= 0:
                    raise ValueError("Модуль упругости должен быть больше нуля.")

                if start_node < 0 or start_node >= len(self.nodes) or end_node < 0 or end_node >= len(self.nodes) or \
                        self.check_rod(start_node, end_node):
                    raise ValueError("Неверные индексы узлов!")

                # Вычисляем длину стержня
                node1 = self.nodes[start_node]
                node2 = self.nodes[end_node]
                length = math.sqrt((node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2)

                # Проверка на положительную длину стержня
                if length <= 0:
                    raise ValueError("Длина стержня должна быть больше нуля.")

                # Обновляем данные существующего стержня
                if rod_index < len(self.rods):  # Если стержень существует
                    rod = self.rods[rod_index]
                    rod.A = area
                    rod.E = material
                    rod.q_p = distributed_load
                    rod.start_node = start_node
                    rod.end_node = end_node
                    rod.L = length
                else:  # Если это новый стержень
                    self.rods.append(Rod(start_node, end_node, material, area, length, distributed_load))

                self.repaint_canvas()
            except ValueError as e:
                QMessageBox.warning(self, "Ошибка", str(e))

        save_rod_button = QPushButton("Сохранить стержень")
        save_rod_button.clicked.connect(update_rod_data)
        self.rod_grid.addWidget(save_rod_button, row, 12)

    def add_node_row(self):
        row = self.node_grid.rowCount()
        x_input = QLineEdit()
        y_input = QLineEdit()
        force_input = QLineEdit()
        fixed_checkbox = QComboBox()
        fixed_checkbox.addItems(["False", "True"])

        self.node_grid.addWidget(QLabel(f"Узел "), row, 0)
        self.node_grid.addWidget(QLabel(f"{row}:"), row, 1)
        self.node_grid.addWidget(QLabel("X:"), row, 2)
        self.node_grid.addWidget(x_input, row, 3)
        self.node_grid.addWidget(QLabel("Y:"), row, 4)
        self.node_grid.addWidget(y_input, row, 5)
        self.node_grid.addWidget(QLabel("Сила:"), row, 6)
        self.node_grid.addWidget(force_input, row, 7)
        self.node_grid.addWidget(QLabel("Закреплён:"), row, 8)
        self.node_grid.addWidget(fixed_checkbox, row, 9)

        # Ищем узел по индексу и обновляем его данные
        def update_node_data():
            try:
                x = float(x_input.text())
                y = float(y_input.text())
                force = float(force_input.text())
                fixed = fixed_checkbox.currentText() == "True"
                # print(self.node_grid.itemAtPosition(row, 1).widget().text()[:-1])
                node_index = int(self.node_grid.itemAtPosition(row, 1).widget().text()[:-1]) - 1

                # Проверка на положительные значения для координат
                if x < 0:
                    raise ValueError("Координата X должна быть больше или равна нулю.")
                if y < 0:
                    raise ValueError("Координата Y должна быть больше или равна нулю.")

                # Обновляем данные существующего узла
                if node_index < len(self.nodes):  # Если узел существует
                    node = self.nodes[node_index]
                    node.x = x
                    node.y = y
                    node.force = force
                    node.fixed = fixed
                else:  # Если это новый узел
                    self.nodes.append(Node(x, y, force, fixed))

                self.repaint_canvas()
            except ValueError as e:
                QMessageBox.warning(self, "Ошибка", str(e))

        save_node_button = QPushButton("Сохранить узел")
        save_node_button.clicked.connect(update_node_data)
        self.node_grid.addWidget(save_node_button, row, 10)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale_factor *= 1.1  # Увеличиваем масштаб
        else:
            self.scale_factor /= 1.1  # Уменьшаем масштаб
        self.repaint_canvas()

    def repaint_canvas(self):
        if not self.nodes:
            return

        # Создание пустого изображения
        pixmap = QPixmap(self.canvas.size())
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)

        # Получаем размеры холста
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()

        # Находим границы конструкции (минимальные и максимальные координаты узлов)
        min_x = min(node.x for node in self.nodes)
        max_x = max(node.x for node in self.nodes)
        min_y = min(node.y for node in self.nodes)
        max_y = max(node.y for node in self.nodes)

        # Вычисляем масштаб для вписывания конструкции в холст с отступами
        padding = 50  # Отступ от краев холста
        structure_width = max_x - min_x if max_x != min_x else 1
        structure_height = max_y - min_y if max_y != min_y else 1

        scale_x = (canvas_width - 2 * padding) / structure_width
        scale_y = (canvas_height - 2 * padding) / structure_height
        scale = min(scale_x, scale_y)

        # Центрирование конструкции
        offset_x = (canvas_width - scale * structure_width) / 2 - min_x * scale
        offset_y = (canvas_height - scale * structure_height) / 2 - min_y * scale

        # Константа для длины стрелки (масштабируемая)
        ARROW_LENGTH = 30

        # Размеры узлов
        NODE_SIZE = 24  # Размер для всех узлов

        # Рисуем стержни
        painter.setPen(QPen(Qt.black, 2))
        for rod in self.rods:
            start = self.nodes[rod.start_node]
            end = self.nodes[rod.end_node]

            # Масштабируем и трансформируем координаты
            start_x = int(start.x * scale + offset_x)
            start_y = int(start.y * scale + offset_y)
            end_x = int(end.x * scale + offset_x)
            end_y = int(end.y * scale + offset_y)

            painter.drawLine(start_x, start_y, end_x, end_y)

        # Рисуем узлы и силы
        for node in self.nodes:
            x = int(node.x * scale + offset_x)
            y = int(node.y * scale + offset_y)

            if node.fixed:
                painter.setBrush(Qt.red)
                painter.setPen(Qt.NoPen)  # Убираем обводку
                painter.drawRoundedRect(x - NODE_SIZE // 2, y - NODE_SIZE // 2, NODE_SIZE, NODE_SIZE, 3, 3)
            else:
                painter.setBrush(Qt.blue)
                painter.setPen(Qt.NoPen)  # Убираем обводку
                painter.drawEllipse(x - NODE_SIZE // 2, y - NODE_SIZE // 2, NODE_SIZE, NODE_SIZE)

            # Рисуем стрелку силы, если сила ненулевая
            if node.force != 0:
                force_x = node.force if isinstance(node.force, (int, float)) else node.force[0]
                force_y = 0 if isinstance(node.force, (int, float)) else node.force[1]
                magnitude = math.sqrt(force_x ** 2 + force_y ** 2) or 1
                direction_x = force_x / magnitude
                direction_y = force_y / magnitude

                end_x = x + int(direction_x * ARROW_LENGTH)
                end_y = y + int(direction_y * ARROW_LENGTH)

                painter.setPen(QPen(Qt.green, 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(x, y, end_x, end_y)

        painter.end()
        self.canvas.setPixmap(pixmap)

    def check_props(self):
        for node in self.nodes:
            if node.fixed is True:
                return False
        return True

    def run_processor(self):
        print("Запуск процесса расчёта...")

        forces = {i: node.force for i, node in enumerate(self.nodes)}
        num_nodes = len(self.nodes)

        if not self.rods:
            QMessageBox.critical(self, "Ошибка", "Не добавлено ни одного стержня!")
            return -1

        if not self.nodes:
            QMessageBox.critical(self, "Ошибка", "Не добавлено ни одного узла!")
            return -1

        if self.check_props():
            QMessageBox.critical(self, "Ошибка", "Не добавлено ни одной опоры!")
            return -1

        # Создание displacements_bc для фиксированных узлов
        displacements_bc = {i: 0.0 for i, node in enumerate(self.nodes) if node.fixed}

        processor = Processor(self.rods, num_nodes, forces, displacements_bc)

        try:
            displacements = processor.solve()
            if displacements is None:
                QMessageBox.critical(self, "Ошибка", "Система уравнений не имеет единственного решения.")
                return -1

            results = processor.get_results()

            if results is None:
                QMessageBox.critical(self, "Ошибка", "Система уравнений не имеет единственного решения.")
                return -1

            displacements, forces, stresses = results

            self.postprocessor = Postprocessor(processor, displacements, forces, stresses)
            self.postprocessor.show()
            self.postprocessor.raise_()
            self.postprocessor.activateWindow()

            self.postprocessor.display_results()

            print(f"Попытка отобразить постпроцессор: {self.postprocessor.isVisible()}")

            displacement_output = "\n".join([f"Узел {i + 1}: ux = {ux:.4f}" for i, ux in enumerate(displacements)])
            force_output = "\n".join([f"Стержень {i + 1}: Nx = [{Nx[0]:.4f}, {Nx[1]:.4f}]" for i, Nx in enumerate(forces)])
            stress_output = "\n".join([f"Стержень {i + 1}: σx = [{stress[0]:.4f}, {stress[1]:.4f}]" for i, stress in enumerate(stresses)])

            result_text = f"Перемещения:\n{displacement_output}\n\nПродольные силы:\n{force_output}\n\nНормальные напряжения:\n{stress_output}"
            self.postprocessor.textEdit.setText(result_text)
            print(result_text)  # Отладочный вывод в консоль
            self.canvas.setText(result_text)  # Отображаем результат на canvas
        except Exception as e:
            print(f"Ошибка при расчете: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при расчете: {str(e)}")
        finally:
            if hasattr(self.postprocessor, 'show') and callable(self.postprocessor.show):
                self.postprocessor.show()

    def load_project_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Загрузить проект", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as file:
                    project_data = json.load(file)

                # Очистка текущих данных
                self.nodes = []
                self.rods = []

                # Загрузка узлов
                for node_data in project_data.get('nodes', []):
                    node = Node(
                        x=node_data['x'],
                        y=node_data['y'],
                        force=node_data.get('force', 0),
                        fixed=node_data.get('fixed', False)
                    )
                    self.nodes.append(node)

                # Загрузка стержней
                for rod_data in project_data.get('rods', []):
                    start_node = rod_data['start_node'] - 1
                    end_node = rod_data['end_node'] - 1
                    rod = Rod(
                        start_node=start_node,
                        end_node=end_node,
                        E=rod_data['E'],
                        A=rod_data['A'],
                        L=rod_data['L'],
                        q_p=rod_data.get('q_p', 0)
                    )
                    self.rods.append(rod)

                # Обновляем интерфейс

                self.repaint_canvas()

                QMessageBox.information(self, "Успех", "Проект успешно загружен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке проекта: {str(e)}")

    def create_project_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "", "JSON Files (*.json)")
        if file_name:
            try:
                project_data = {
                    'nodes': [{'x': node.x, 'y': node.y, 'force': node.force, 'fixed': node.fixed} for node in
                              self.nodes],
                    'rods': [{
                        'E': rod.E, 'A': rod.A, 'L': rod.L, 'q_p': rod.q_p,
                        'start_node': rod.start_node + 1, 'end_node': rod.end_node + 1
                    } for rod in self.rods]
                }

                with open(file_name, 'w') as file:
                    json.dump(project_data, file, indent=4)

                QMessageBox.information(self, "Успех", "Проект успешно сохранён.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении проекта: {str(e)}")


class Postprocessor(QWidget):
    def __init__(self, processor, displacements, forces, stresses):
        super().__init__()
        self.processor = processor
        self.displacements = displacements
        self.forces = forces
        self.stresses = stresses
        self.initUI()

    def get_processor_results(self):
        return self.processor.get_results()

    def initUI(self):
        self.setWindowTitle("Постпроцессор результатов расчёта")

        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Postprocessor')

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Результаты расчёта:"))
        self.textEdit = QTextEdit()
        layout.addWidget(self.textEdit)
        self.setLayout(layout)

        # Кнопка для отображения текстовых результатов
        self.results_button = QPushButton("Показать текстовые результаты")
        self.results_button.clicked.connect(self.display_results)
        layout.addWidget(self.results_button)

        # Кнопка для отображения таблицы результатов
        self.table_button = QPushButton("Показать таблицу результатов")
        self.table_button.clicked.connect(self.display_table)
        layout.addWidget(self.table_button)

        # Кнопка для построения графиков
        self.plot_button = QPushButton("Построить графики результатов")
        self.plot_button.clicked.connect(self.plot_results)
        layout.addWidget(self.plot_button)

        self.save_button = QPushButton("Сохранить результаты в файл")
        self.save_button.clicked.connect(self.save_results_to_file)
        layout.addWidget(self.save_button)

        self.analyze_button = QPushButton("Анализ результатов")
        self.analyze_button.clicked.connect(self.analyze_results)
        layout.addWidget(self.analyze_button)

        # Поле для ввода номера стержня
        self.rod_input = QLineEdit()
        self.rod_input.setPlaceholderText("Введите номер стержня")

        # Поле для ввода соотношения длины (от 0 до 1)
        self.section_ratio_input = QLineEdit()
        self.section_ratio_input.setPlaceholderText("Введите относительное положение (0-1)")

        # Кнопка для получения результатов в сечении
        self.section_button = QPushButton("Показать результаты в сечении")
        self.section_button.clicked.connect(self.handle_section_results)

        # Добавляем виджеты в макет
        layout.addWidget(QLabel("Номер стержня:"))
        layout.addWidget(self.rod_input)
        layout.addWidget(QLabel("Относительное положение (0-1):"))
        layout.addWidget(self.section_ratio_input)
        layout.addWidget(self.section_button)

    def handle_section_results(self):
        try:
            rod_index = int(self.rod_input.text()) - 1
            section_ratio = float(self.section_ratio_input.text())

            # Проверка корректности ввода
            if rod_index < 0 or rod_index >= len(self.processor.rods):
                QMessageBox.warning(self, "Ошибка", "Неверный номер стержня.")
                return

            if section_ratio < 0 or section_ratio > 1:
                QMessageBox.warning(self, "Ошибка", "Относительное положение должно быть между 0 и 1.")
                return

            # Вызов функции для получения результатов
            self.get_section_results(rod_index, section_ratio)

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные числовые значения.")

    def save_results_to_file(self):
        try:
            # Получаем текстовые результаты расчёта
            displacement_output = "\n".join([f"Узел {i + 1}: ux = {ux:.4f}" for i, ux in enumerate(self.displacements)])
            force_output = "\n".join([f"Стержень {i + 1}: Nx = [{Nx[0]:.4f}; {Nx[1]:.4f}]" for i, Nx in enumerate(self.forces)])
            stress_output = "\n".join([f"Стержень {i + 1}: \u03c3x = [{stress[0]:.4f}; {stress[1]:.4f}]" for i, stress in enumerate(self.stresses)])

            result_text = f"Перемещения:\n{displacement_output}\n\nПродольные силы:\n{force_output}\n\nНормальные напряжения:\n{stress_output}"

            # Открываем диалог сохранения файла
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить результаты", "",
                                                       "Text Files (*.txt);;All Files (*)", options=options)

            # Если пользователь выбрал файл, записываем результаты в файл
            if file_path:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(result_text)
                QMessageBox.information(self, "Успех", f"Результаты сохранены в файл: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты: {e}")

    def analyze_results(self):
        max_displacement = max(self.displacements)
        min_displacement = min(self.displacements)

        max_force = max(self.forces[0])
        min_force = min(self.forces[1])

        max_stress = max(self.stresses[0])
        min_stress = min(self.stresses[1])

        analysis_text = (
            f"Максимальное перемещение: {max_displacement:.4f}\n"
            f"Минимальное перемещение: {min_displacement:.4f}\n\n"
            f"Максимальная сила: {max_force:.4f}\n"
            f"Минимальная сила: {min_force:.4f}\n\n"
            f"Максимальное напряжение: {max_stress:.4f}\n"
            f"Минимальное напряжение: {min_stress:.4f}"
        )

        QMessageBox.information(self, "Анализ результатов", analysis_text)

    def get_section_results(self, rod_index, section_ratio):
        rod = self.processor.rods[rod_index]
        local_coord = rod.L * section_ratio

        stress = self.stresses[rod_index]
        force = self.forces[rod_index]
        disp_start = self.displacements[rod.start_node]
        disp_end = self.displacements[rod.end_node]
        displacement = disp_start + (disp_end - disp_start) * section_ratio

        section_text = (
            f"Стержень {rod_index + 1}, Сечение на {section_ratio * 100:.0f}% длины:\n"
            f"Перемещение: {displacement:.4f}\n"
            f"Сила: [{force[0]:.4f}; {force[1]:.4f}]\n"
            f"Напряжение: [{stress[0]:.4f}; {stress[1]:.4f}]"
        )

        QMessageBox.information(self, "Результаты в сечении", section_text)

    def display_results(self):
        displacement_output = "\n".join([f"Узел {i + 1}: ux = {ux:.4f}" for i, ux in enumerate(self.displacements)])
        force_output = "\n".join([f"Стержень {i + 1}: Nx = [{Nx[0]:.4f}, {Nx[1]:.4f}]" for i, Nx in enumerate(self.forces)])
        stress_output = "\n".join(
            [f"Стержень {i + 1}: \u03c3x = [{stress[0]:.4f}; {stress[1]:.4f}]" for i, stress in enumerate(self.stresses)])
        result_text = f"Перемещени:\n{displacement_output}\n\nПродольные силы:\n{force_output}\n\nНормальные напряжения:\n{stress_output}"
        self.textEdit.setText(result_text)

    def display_table(self, rod_index):
        # Создаём таблицу с числом строк, равным количеству стержней и 7 столбцами
        table = QTableWidget(len(self.processor.rods), 7)
        table.setHorizontalHeaderLabels([
            "Стержень",
            "Nx (Сила начало)", "Nx (Сила конец)",
            "\u03c3x (Напряжение начало)", "\u03c3x (Напряжение конец)",
            "Ux (Перемещение начало)", "Ux (Перемещение конец)"
        ])

        rod = self.processor.rods[rod_index]
        disp_start = self.displacements[rod.start_node]
        disp_end = self.displacements[rod.end_node]

        for i, (force, stress, displacements) in enumerate(zip(self.forces, self.stresses, self.displacements)):
            # Номер стержня
            table.setItem(i, 0, QTableWidgetItem(f"{i + 1}"))

            # Силы Nx
            table.setItem(i, 1, QTableWidgetItem(f"{force[0]:.4f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{force[1]:.4f}"))

            # Напряжения σx
            table.setItem(i, 3, QTableWidgetItem(f"{stress[0]:.4f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{stress[1]:.4f}"))

            # Перемещения Ux
            table.setItem(i, 5, QTableWidgetItem(f"{disp_start:.4f}"))
            table.setItem(i, 6, QTableWidgetItem(f"{disp_end:.4f}"))

        self.table_window = QWidget()
        self.table_window.setWindowTitle("Таблица результатов")
        self.table_window.setGeometry(350, 350, 700, 300)

        layout = QVBoxLayout()
        layout.addWidget(table)
        self.table_window.setLayout(layout)

        self.table_window.show()

    @staticmethod
    def parabola_points_with_y0(start, stop, num=50, a=1, b=0, y0=0):
        x1 = np.linspace(start, y0, num//2)
        x2 = np.linspace(y0, stop, num//2)
        x = np.append(x1, x2)

        c = y0
        y = a * x ** 2 + b * x + c

        return x, y

    @staticmethod
    def find_y0_n_x(rod, displacement_end, displacement_start):
        y0 = (((rod.E * rod.A) * (displacement_end - displacement_start) / rod.L * (rod.q_p * rod.L / 2) + 1) * rod.L) / 2
        if rod.q_p >= 0:
            return y0
        else:
            return -1 * y0

    def plot_results(self):
        rods = self.processor.rods
        num_rods = len(rods)

        # Локальные координаты для построения эпюр (100 точек на каждый стержень)
        local_coords = [np.linspace(0, rod.L, 100) for rod in rods]

        # Сбор данных для построения эпюр
        stresses_epures = []
        forces_epures = []
        displacements_epures = []

        for i, rod in enumerate(rods):
            # Напряжения (линейное распределение)
            sigma_start = self.stresses[i][0]
            sigma_end = self.stresses[i][1]
            stresses_epures.append(np.linspace(sigma_start, sigma_end, 100))

            # Силы (линейное распределение)
            force_start = self.forces[i][0]
            force_end = self.forces[i][1]
            forces_epures.append(np.linspace(force_start, force_end, 100))

            # Перемещения (линейное распределение между узлами)
            disp_start = self.displacements[rod.start_node]
            disp_end = self.displacements[rod.end_node]
            # dotes_of_epure = np.linspace(disp_start, disp_end, 100)

            # Если с распределительной нагрузкой, то парабола
            # N_x = rod.E * rod.A / rod.L * (displacement_end - displacement_start) + (rod.q_p * rod.L / 2) * (1 - 2 * x / rod.L)
            y0 = self.find_y0_n_x(rod, disp_end, disp_start)
            dotes_of_epure = self.parabola_points_with_y0(disp_start, disp_end, 100, y0=y0)
            displacements_epures.append(dotes_of_epure)

        # Создание фигуры с тремя строками и `num_rods` столбцами
        fig, axs = plt.subplots(nrows=3, ncols=num_rods, figsize=(4 * num_rods, 12), constrained_layout=True)

        # Если один стержень, приводим axs к двумерному массиву
        if num_rods == 1:
            axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

        # Цвета и подписи для графиков
        colors = ['purple', 'green', 'blue']
        titles = ['Силы $N_x$', 'Перемещения $u_x$', 'Напряжения $\sigma_x$']
        y_labels = [r'$N_x$', r'$u_x$', r'$\sigma_x$']

        for i in range(num_rods):
            # График сил для текущего стержня
            axs[0, i].fill_between(local_coords[i], 0, forces_epures[i], color=colors[0], alpha=0.3)
            axs[0, i].plot(local_coords[i], forces_epures[i], color=colors[0], linewidth=1.5)
            axs[0, i].set_title(f"{titles[0]} в стержне {i + 1}")
            axs[0, i].set_ylabel(y_labels[0])
            axs[0, i].grid(True)

            # График перемещения для текущего стержня
            axs[1, i].plot(local_coords[i], displacements_epures[i][0], color=colors[1], linewidth=1.5)
            axs[1, i].set_title(f"{titles[1]} в стержне {i + 1}")
            axs[1, i].set_ylabel(y_labels[1])
            axs[1, i].grid(True)

            # График напряжений для текущего стержня
            axs[2, i].fill_between(local_coords[i], 0, stresses_epures[i], color=colors[2], alpha=0.3)
            axs[2, i].plot(local_coords[i], stresses_epures[i], color=colors[2], linewidth=1.5)
            axs[2, i].set_title(f"{titles[2]} в стержне {i + 1}")
            axs[2, i].set_ylabel(y_labels[2])
            axs[2, i].grid(True)

        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Preprocessor()
    ex.show()
    sys.exit(app.exec_())
