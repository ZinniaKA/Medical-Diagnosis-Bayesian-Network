import time
import sys

class Graph_Node():


    def __init__(self, name, n, values) -> None:
        self.Node_Name = name
        self.no_of_values = n
        self.values = values

        self.Children = [] # List of children node indexes
        self.Parents = [] #List of Parent node indexes
        self.Parent_names = [] #List of Parent node names
        self.CPT = []

    def get_name(self):
        return self.Node_Name
    
    def get_children(self):
        return self.Children
    
    def get_parents(self):
        return self.Parents
    
    def get_CPT(self):
        return self.CPT
    
    def get_values(self):
        return self.values
    
    def get_no_of_values(self):
        return self.no_of_values
    
    def set_CPT(self, new_CPT):
        self.CPT = new_CPT

    #set parents and set children is not needed I think

    def add_child(self, child):
        if child not in self.Children:
            self.Children.append(child)
    

class Network:
    def __init__(self):
        self.Graph_Network = []  # Initialize the Graph_Network attribute within the constructor
        self.NodetoIndexHash = {} # Hash table to store the index of the node name in the Graph_Network
        self.IndextoNodeHash = {} # Hash table to store the node name given the index in the Graph_Network
        self.matrixValues = []   # Matrix to store the values of the nodes in the network
        self.matrixParents = []  #index of parents
        self.matrixCPT = [] # Matrix to store the CPT of the nodes in the network
        self.matrixChildren = []
        self.nvalues = []
        self.datapoints = []
    
    def add_node(self, node):
        self.Graph_Network.append(node)
        self.NodetoIndexHash[node.get_name()] = len(self.Graph_Network) - 1
        self.IndextoNodeHash[len(self.Graph_Network) - 1] = node.get_name()
    
    def net_size(self):
        return len(self.Graph_Network)
    
    def get_node_state(self, index, cur_val):
        for i in range(len(self.matrixValues[index])):
            val = self.Graph_Network[index].values[i]
            if val == cur_val:
                return i
    
    def get_node(self, index):
        return self.Graph_Network[index]
    
    def search_node(self, name):
        if name in self.NodetoIndexHash:
            return self.Graph_Network[self.NodetoIndexHash[name]]
        return None
    
    def get_node_index(self, name):
        if name in self.NodetoIndexHash:
            return self.NodetoIndexHash[name]
        return -1

    def write_to_file(self, filename):
        with open(filename, 'w') as file:
            for node in self.Graph_Network:
                file.write(f"Node Name: {node.get_name()}\n")
                # # You can add more information about the node here
                # # For example, you can write its children, parents, CPT, etc.
                # file.write(f"Parents: {node.get_parents()}\n")
                # # Add more information as needed
                # file.write(f"CPT: {node.get_CPT()}\n")
                # file.write("\n")
            file.write(f"matrixChildren: {self.matrixChildren}\n")
            # file.write(f"matrixValues: {self.matrixValues}\n")
            file.write (f"matrixParents: {self.matrixParents}\n")
            file.write(f"nvalues: {self.nvalues}\n")
            file.write(f"matrixCPT: {self.matrixCPT}\n")

    # def compute_nvals(self):
    #     for i in range(len(self.Graph_Network)):
    #         self.nvalues.append(self.Graph_Network[i].get_no_of_values())




def read_network():
    Alarm = Network()
    find = 0
    node_index = 0

    with open(networkfilename, 'r') as myfile:
        no_of_nodes_found = 0
        no_of_nodes_found2 = 0
        for line in myfile:
            temp = ''
            name = ''   #name of the node
            if line.strip() == "":
                continue

            after_splitting_line = line.split()

            if after_splitting_line[0] == "variable":
                values = []
                no_of_nodes_found += 1
                name = after_splitting_line[1]
                line = next(myfile)
                values_line_lis = line.split()
                for i in range(3, len(values_line_lis)):
                    if values_line_lis[i] == "};":
                        break
                    else:
                        values.append(values_line_lis[i])
                Alarm.nvalues.append(len(values))
                # if no_of_nodes_found < 3:
                    # print("name :", name, "values :", values, "length :", len(values))
                Alarm.matrixValues.append(values)
                new_node = Graph_Node(name, len(values), values)
                Alarm.add_node(new_node)
                Alarm.matrixParents.append([])
                Alarm.matrixChildren.append([])

            elif after_splitting_line[0] == "probability":
                no_of_nodes_found2 += 1
                parents = []
                cpt = []
                cpt_len = 0
                name = after_splitting_line[1]
                cur_node_index = Alarm.get_node_index(after_splitting_line[2])
                for i in range (3, len(after_splitting_line)):
                    if after_splitting_line[i] == ")":
                        break
                    else:
                        parent_index = Alarm.get_node_index(after_splitting_line[i])
                        # parent_index = Alarm.get_node_index(parent_name)
                        Alarm.matrixParents[cur_node_index].append(parent_index)
                        Alarm.Graph_Network[parent_index].add_child(cur_node_index)
                        Alarm.matrixChildren[parent_index].append(cur_node_index)
                        # Alarm.add_child(child)
                        parents.append(parent_index)
                        
                line = next(myfile)
                probabilities_line_lis = line.split()

                for i in range(1, len(probabilities_line_lis)):
                    if probabilities_line_lis[i] == ";":
                        break
                    else:
                        cpt_len += 1
                cpt = []
                for i in range(cpt_len):
                    cpt.append(1/Alarm.nvalues[cur_node_index])
                    
                # cpt = [1/cpt_len] * cpt_len
                # if no_of_nodes_found2 < 3:
                    # print("name :", name, "parents :", parents, "cpt :", cpt, "length :", len(cpt))
                Alarm.matrixCPT.append(cpt)
                Alarm.Graph_Network[cur_node_index].set_CPT(cpt)
                Alarm.Graph_Network[cur_node_index].Parents = parents
                node_index += 1

        myfile.close()
    return Alarm


def read_dataset(Alarm, q_indexes):
    dataset = []
    no_of_lines = 0
    missing = -1
    with open(recordsfilename, 'r') as myfile:
        for line in myfile:
            no_of_lines += 1
            foundq = False
            if line.strip() == "":
                continue
            data_point = line.split()
            int_data_point = []
            for i in range(len(data_point)):
                if data_point[i] == '"?"':
                    foundq = True
                    missing = i
                    int_data_point.append(-1)
                    # q_indexes.append(i)
                else:
                    int_data_point.append(Alarm.get_node_state(i, data_point[i]))

            if foundq:
                for z in range (Alarm1.nvalues[missing]):
                    new_int_data_point = int_data_point.copy()
                    new_int_data_point[missing] = z
                    Alarm1.datapoints.append(new_int_data_point)


            dataset.append(int_data_point)
            q_indexes.append(missing)

    return dataset


def get_CPT_index(each_node_state, sizes):
    if (len(each_node_state) == 0):
        return 0
    index = 0
    b = 1
    for i in range(len(each_node_state) - 1, -1, -1):
        index += b * each_node_state[i]
        b *= sizes[i]
    return index



def expectation(weights):
    weights.clear()
    for i in range(len(data_set)):
        qindex = q_indexes[i]
        if qindex == -1:
            weights.append(1)
            continue
        else:
            node_values = Alarm1.matrixValues[qindex]
            num = 0
            den = 0
            N = Alarm1.nvalues[qindex]
            all_numerators = [] #dont know why this is needed there in utkarsh code
            for t in range(N):
                num = 1
                q_data_point = data_set[i].copy()
                q_data_point[qindex] = t
                maxC = len(Alarm1.matrixChildren[qindex])
                for j in range(0, maxC):
                    vals = []
                    sizes = []
                    vals.append(data_set[i][Alarm1.matrixChildren[qindex][j]])
                    sizes.append(Alarm1.nvalues[Alarm1.matrixChildren[qindex][j]])

                    for k in range (0, len(Alarm1.matrixParents[Alarm1.matrixChildren[qindex][j]])):
                        sizes.append(Alarm1.nvalues[Alarm1.matrixParents[Alarm1.matrixChildren[qindex][j]][k]])
                        vals.append(q_data_point[Alarm1.matrixParents[Alarm1.matrixChildren[qindex][j]][k]])
                    index = get_CPT_index(vals, sizes)
                    num = num*Alarm1.matrixCPT[Alarm1.matrixChildren[qindex][j]][index]
                #den = den + num
                # all_numerators.append(num)
                vals = []
                sizes = []
                vals.append(t)
                sizes.append(N)

                for z in range(0, len(Alarm1.matrixParents[qindex])):
                    vals.append(q_data_point[Alarm1.matrixParents[qindex][z]])
                    sizes.append(Alarm1.nvalues[Alarm1.matrixParents[qindex][z]])
                num = num*Alarm1.matrixCPT[qindex][get_CPT_index(vals, sizes)]
                all_numerators.append(num)
                den = den + num
            for j in range(len(all_numerators)):
                weights.append(all_numerators[j]/den)




def maximisation():
    max_diff = 0
    for i in range (0, len(Alarm1.matrixCPT)):
        vals = []
        sizes = []
        vals_index = []
        vals_index.append(i)
        sizes.append(Alarm1.nvalues[i])
        for j in range (len(Alarm1.matrixParents[i])):
            vals_index.append(Alarm1.matrixParents[i][j])
            sizes.append(Alarm1.nvalues[Alarm1.matrixParents[i][j]])
        MODDA = int(len(Alarm1.matrixCPT[i])/Alarm1.nvalues[i])
        # print("MODDA :", MODDA)
        denominators = [0]*MODDA
        numerators = [0]*len(Alarm1.matrixCPT[i])
        for j in range (len(Alarm1.datapoints)):
            vals = []
            for k in range (len(vals_index)):
                vals.append(Alarm1.datapoints[j][vals_index[k]])
            index = get_CPT_index(vals, sizes)
            numerators[index] = numerators[index] + weights[j]
            denominators[index%MODDA] = denominators[index%MODDA] + weights[j]
        
        for j in range (len(Alarm1.matrixCPT[i])):
            temp = (numerators[j]+0.001)/(denominators[j%MODDA]+0.001*Alarm1.nvalues[i])
            max_diff = max(max_diff, abs(temp-Alarm1.matrixCPT[i][j]))
            Alarm1.matrixCPT[i][j] = temp
            if Alarm1.matrixCPT[i][j]<0.01:
                Alarm1.matrixCPT[i][j] = 0.01
    
def write_network(filename):
    with open(filename, 'r') as myfile, open('solved_alarm.bif', 'w') as outfile:
        counter = 0
        for line in myfile:
            temp = line.split()[0]
            if temp == "probability":
                outfile.write(line)
                line = next(myfile)
                cpt_str = " ".join([f"{x:.4f}" for x in Alarm1.matrixCPT[counter]])  # Format CPT values
                # outfile.write(f"  table {cpt_str};\n")
                outfile.write(f"\ttable {cpt_str} ;\n")
                counter += 1
            else:
                if line.strip():
                    outfile.write(line)
                else:
                    outfile.write(line)




if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("insufficient arguments")
        sys.exit(1)

    networkfilename = sys.argv[1]
    recordsfilename = sys.argv[2]

    q_indexes = []
    Alarm1 = read_network()

    data_set = read_dataset(Alarm1, q_indexes)
    weights = []

    print(networkfilename, recordsfilename)
    start_time = time.time()
    while True:
        if time.time()-start_time>110:
            break
        expectation(weights)
        maximisation()
    write_network(networkfilename)








# print(Alarm1.matrixCPT)

    

    