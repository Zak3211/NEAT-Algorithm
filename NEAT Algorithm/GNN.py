from collections import defaultdict
from collections import deque
import random
import math
import copy
import functools

class GraphNode:
    def __init__(self, id):
        self.id = id
        self.value = 0
        self.bias = random.randint(-100,100)/100

class graphNeuralNetwork: 
    def __init__(self, numInputs = 5, numOutputs = 1):
        """Creates a Graph Neural Network with Tensorized Paramters"""

        #Initializes object paramters
        self.idToNode = {}
        self.currId = 0

        self.ids = []
        self.idSet = set()

        self.graph = defaultdict(list)
        self.reverseGraph = defaultdict(list)

        self.connections = {}
        self.connectionList = []
        
        #Initializes input, output, and internal nodes.
        self.inputNodes = [self.addInitialNode(isInner=False) for _ in range(numInputs)]
        self.outputNodes = [self.addInitialNode(isInner=False) for _ in range(numOutputs)]
        self.internalNodes = [self.addInitialNode(isInner=True) for _ in range(6)]

        #Stores input and output ids in a set for quick lookup.
        self.inputSet = set(self.inputNodes)
        self.outputSet = set(self.outputNodes)


    def addInitialNode(self, isInner):
        """Adds a node connected to an input and output node"""
        
        #Creates newNode and updates paramters
        newNode = GraphNode(self.currId)
        self.idToNode[self.currId] = newNode
        self.ids.append(self.currId)
        self.idSet.add(self.currId)

        if not isInner:
            self.currId += 1
            return self.currId - 1
        
        #Selects a random input and output node
        inputNode = random.choice(self.inputNodes)
        outputNode = random.choice(self.outputNodes)

        #inputNode => newNode connection
        self.connectionList.append((inputNode, newNode.id))
        self.connections[(inputNode, newNode.id)] = random.randint(-100, 100)/100
        self.reverseGraph[newNode.id].append(inputNode)
        self.graph[inputNode].append(newNode.id)

        #newNode => outputNode conneciton
        self.connectionList.append((newNode.id, outputNode))
        self.connections[(newNode.id, outputNode)] = random.randint(-100, 100)/100
        self.reverseGraph[newNode.id].append(outputNode)
        self.graph[outputNode].append(newNode.id)

        self.currId += 1
        return self.currId - 1
    
    def addNode(self):
        """Splits a current connection with a node."""

        if not self.connectionList:
            return False
        
        #Deletes a current connection saving prevNode := connection[0], nextNode := connection[1]
        connectionIndex = random.randint(0, len(self.connectionList)-1)
        connection = self.connectionList[connectionIndex]
        connectionWeight = self.connections[connection]
        del self.connections[connection]
        self.connectionList.pop(connectionIndex)

        #Initializes a new node and updates relevant paramters
        newNode = GraphNode(self.currId)
        self.idToNode[self.currId] = newNode
        self.ids.append(self.currId)
        self.idSet.add(self.currId)
        self.internalNodes.append(self.currId)

        #Adds the first connection: prevNode => newNode
        self.connectionList.append((connection[0], self.currId))
        self.connections[(connection[0], self.currId)] = connectionWeight
        self.reverseGraph[self.currId].append(connection[0])
        self.graph[connection[0]].append(self.currId)

        #Adds the second connection: newNode => nextNode
        self.connectionList.append((self.currId, connection[1]))
        self.connections[(self.currId, connection[1])] = random.randint(-100,100)/100
        self.reverseGraph[connection[1]].append(self.currId)
        self.graph[self.currId].append(connection[1])

        self.currId += 1
        return True
    
    def deleteNode(self):
        """Selects a random inner node and deletes it entirely"""

        #Empty inner nodes edge case
        if not self.internalNodes:
            return False

        #Selects an element of self.internalNodes and deletes it.
        nodeIndex = random.randint(0, len(self.internalNodes)-1)
        self.internalNodes.pop(nodeIndex)

        #Updates index to match with self.ids
        nodeIndex += len(self.inputNodes) + len(self.outputNodes)

        #Removes the node from self.ids
        nodeId = self.ids[nodeIndex]
        self.ids.pop(nodeIndex)
        self.idSet.remove(nodeId)

        return True

    def addConnection(self):
        """Selects two random nodes and adds a connection between them if valid."""
            
        #Selects two random nodes
        node1 = random.choice(self.ids)
        node2 = random.choice(self.ids)

        #Checks validity of nodes
        if node1 == node2:
            return False
        if node1 in self.outputSet or node2 in self.inputSet:
            return False
        if (node1, node2) in self.connections:
            return False

        #Adds a phantom connection and checks for loops
        self.graph[node1].append(node2)
        if self.detectLoop(node1):
            self.graph[node1].pop()
            return False
            
        #Updates paramters if there are no issues
        self.reverseGraph[node2].append(node1)
        self.connections[(node1, node2)] = random.randint(-100,100)/100
        self.connectionList.append((node1, node2))

        return True
    
    def removeConnection(self):
        """Removes a connection if there are any."""

        #Handling no connections edge case
        if not self.connectionList:
            return False
        
        #Selects a random connection and deletes it
        connectionIndex = random.randint(0, len(self.connectionList)-1)
        del self.connections[self.connectionList[connectionIndex]]
        self.connectionList.pop(connectionIndex)

        return True        

    def detectLoop(self, nodeId):
        """Basic Breadth-First Search using a Queue to detect a loop."""

        visited = set()
        queue = deque([nodeId])
        while queue:
            currNode = queue.popleft()
            if currNode in visited:
                return True
            visited.add(currNode)
            for neighbour in self.graph[currNode]:
                queue.append(neighbour)
        return False

    def setInputs(self, inputs):
        """"Sets the value of input nodes to the input parameter."""

        #Iterates over inputs and assigns node values.
        for i in range(len(inputs)):
            self.idToNode[self.inputNodes[i]].value = inputs[i]

    @functools.lru_cache(None)
    def getNodeValue(self, nodeId):
        """Basic dfs that self.reverseGraph and retrieves inputs"""

        #Delted node edge case
        if nodeId not in self.idSet:
            return 0
        
        currNode = self.idToNode[nodeId]

        #Base case for recursive function.
        if nodeId in self.inputSet:
            return currNode.value

        currNode.value = 0
        for neighbor in self.reverseGraph[nodeId]:

            #Delted connection edge case
            if (neighbor, nodeId) not in self.connections:
                continue

            currNode.value += self.connections[(neighbor, nodeId)]*self.getNodeValue(neighbor)

        #Apply activation function
        currNode.value = math.tanh(currNode.value + currNode.bias) 

        return currNode.value
    
    def forward(self, inputs):
        """Intializes the inputs and runs dfs on all output nodes."""

        #Resets the cache
        self.getNodeValue.cache_clear()

        #Prepares inputs and initializes output array
        self.setInputs(inputs)
        outPutLayer = [self.getNodeValue(outputNode) for outputNode in self.outputNodes]

        #Edge case for one output
        if len(outPutLayer) == 1:
            outPutLayer = outPutLayer[0]

        return 70*outPutLayer

    def mutate(self):
        """Creates a mutated copy of the current network."""

        #Creates a deepcopy of the current network
        newNetwork = copy.deepcopy(self)

        #Pertubates weights and biases of child
        for connection in newNetwork.connections.keys():
            newNetwork.connections[connection] += random.randint(-100,100)/1000
        for nodeId in newNetwork.ids:
            newNetwork.idToNode[nodeId].bias += random.randint(-100,100)/1000
        
        #Augments child topology based on basic probability logic
        #There is the option to use more advanced statistical methods for better performance.
        random_variable = random.randint(0,35)
        if random_variable == 1 or random_variable == 2:
            newNetwork.addConnection()
        elif random_variable == 3:
            newNetwork.removeConnection()
        elif random_variable == 4:
            newNetwork.addNode()
        elif random_variable == 5:
            newNetwork.deleteNode()

        return newNetwork
