from PyQt5.QtWidgets import QApplication, QDesktopWidget, QGraphicsScene, QGraphicsItem, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem
from PyQt5.QtGui import QPen, QPainter, QFont
from PyQt5.QtCore import Qt
import sys
import math
import torch

class ResizableTextNode(QGraphicsRectItem):
    def __init__(self, x, y, text, width=120, height=50, padding=8, fontSize=10):
        super().__init__(x, y, width, height)
        self.x = x
        self.y = y
        
        # To draw node text first before lines
        self.setZValue(1)

        # Rect text node graphics
        self.setBrush(Qt.white)
        self.setPen(QPen(Qt.black, 2))
        self.setFlags(QGraphicsRectItem.ItemIsMovable | QGraphicsRectItem.ItemIsSelectable | QGraphicsRectItem.ItemSendsGeometryChanges)
        self.padding = padding
        self.textItem = QGraphicsTextItem(text, self)
        self.textItem.setFont(QFont("Arial", fontSize))
        self.connections = []

        # Drag / move variable initalisation
        self.drag = False
        self.previousPositionX = 0
        self.previousPositionY = 0
        self.minWidth = 70

        # Set initial wrapping width
        self.textItem.setTextWidth(width - 2 * padding)
        self.textItem.setDefaultTextColor(Qt.black)
        self.updateTextPos()

    # Updates text to fit in node bounding rect
    def updateTextPos(self):
        height = self.textItem.boundingRect().height() + 2 * self.padding
        self.setRect(self.x, self.y, self.rect().width(), height)
        self.textItem.setTextWidth(self.rect().width() - 2 * self.padding)
        self.textItem.setPos(self.x + self.padding/2, self.y + self.padding/2)

    # Allows for dragging of nodes and resizing by right clicking and dragging the bottom right corner
    def mousePressEvent(self, event):
        self.drag = False
        rect = self.rect()
        pos = event.pos()
        # Drag resize detection
        resizeMargin = min(rect.width(),rect.height())/4
        if event.button() == Qt.RightButton:
            if pos.x() - self.x > resizeMargin and pos.y() - self.y > resizeMargin:
                self.drag = True
        else:
            super().mousePressEvent(event)
        self.previousPositionX = pos.x()
        self.previousPositionY = pos.y()

    # Handles node dragging and resizing
    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.drag:
            rect = self.rect()
            diff = pos.x()-self.previousPositionX
            # Update rect with new size
            self.setRect(self.x, self.y, max(self.minWidth, rect.width()+diff), rect.height())
            self.updateTextPos()
            self.updateConnections()
        else:
            super().mouseMoveEvent(event)
        self.previousPositionX = pos.x()
        self.previousPositionY = pos.y()    

    # Handle position changing for children connection lines
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.updateConnections()
        return super().itemChange(change, value)

    # Update connected node lines
    def updateConnections(self):
        for connection in self.connections:
            connection.updatePosition()

class ConnectionLine(QGraphicsLineItem):

    def __init__(self, node1, node2):
        super().__init__()
        # Set up line graphics
        self.setPen(QPen(Qt.black, 2))
        # Respective connected nodes
        self.node1 = node1
        self.node2 = node2
        self.translation = [0,0]
        self.setZValue(0)
        self.node1.connections.append(self)
        self.node2.connections.append(self)
        self.updatePosition()

    # Updates the position of the node line connector when text rect is moved
    def updatePosition(self):
        center1 = self.node1.sceneBoundingRect().center()
        center2 = self.node2.sceneBoundingRect().center()    
        self.setLine(center1.x(), center1.y(), center2.x(), center2.y())

# Generates visual window to display flow charts
class FlowchartView(QGraphicsView):
    def __init__(self):
        super().__init__()

        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)  # Enable smooth rendering
        
        # Ensure scroll bars are visible
        self.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOn )
        self.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOn )

        # Set up viewport (maximised)
        viewport = QDesktopWidget().screenGeometry()
        self.setSceneRect(0, 0, viewport.width(), viewport.height())
        self.showMaximized()
        self.windowWidth = self.size().width()
        self.windowHeight = self.size().height()

        # Scrolling parameters
        self.maxScroll = 0.25
        self.scrollDampening = 40

        self.nodes = []  # Store nodes for correct ordering
        self.lines = []  # Store lines

    # Zooming with scroll wheel
    def wheelEvent(self, event):
        delta = 1 - min(-event.angleDelta().y()/self.scrollDampening,self.maxScroll) if event.angleDelta().y() < 0 else 1 + min(event.angleDelta().y()/self.scrollDampening,self.maxScroll)
        self.scale(delta,delta)
    
    # Add node to view
    def addNode(self, x, y, text, width=120, padding=8, fontSize=13):
        node = ResizableTextNode(x, y, text, width=width, padding=padding, fontSize=fontSize)
        self.scene.addItem(node)
        return node

    # Add connection line to view
    def addConnection(self, node1, node2):
        line = ConnectionLine(node1, node2)
        self.scene.addItem(line)

def createNodes(view, tokenizer, x, y, arr, width=100, shiftAmt=500, dropAmt=250, shiftReduction=2, previousNode=None):
    """
    Recursively creates and positions nodes in a tree visualization.
    """
    node_data = arr[0]
    if "confidence" in node_data:
        # Rounds to 2 DP
        info = ("\n\n Confidence: " + str(round(sum(node_data["confidence"])/len(node_data["confidence"]), 2)))
        # Add current node to tree
        currentNode = view.addNode(x, y, tokenizer.decode(node_data["output"], skip_special_tokens=True) + info, width)
    else:
        # Add current node to tree
        currentNode = view.addNode(x, y, tokenizer.decode(node_data["output"], skip_special_tokens=True), width)

    # Add connections to tree
    if previousNode is not None:
        view.addConnection(previousNode, currentNode)

    # Shift amount, depends on #nodes
    if (len(arr)-1) % 2 == 1:
        shift = -math.floor((len(arr)-1)/2) * shiftAmt
    else:
        shift = -(len(arr)-1)/2 * (shiftAmt/2)

    # Recursive visual node generation
    for i in range(1, len(arr)):
        createNodes(view, tokenizer, x+shift, y+(dropAmt/2 if previousNode is None else dropAmt), 
                   arr[i], width, shiftAmt/shiftReduction, dropAmt, shiftReduction, currentNode)
        shift += shiftAmt