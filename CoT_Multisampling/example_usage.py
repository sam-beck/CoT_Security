import multisampling
import visualization
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Choose device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device) # Move to chosen device

# Encode input prompt
prompt = 'If you rearrange the letters in "new door," what two words do you get?'
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device) 

# List of nodal structure for each node in the tree
nodes = [2,2,1] # First node generation has 3 branches, second has 2 branches, third has 1 branch

# Generate output using multisampling and specific inference parameters
output = multisampling.CoTTreeTokens(
    model, input_ids, nodes, True, 
    temperature=0.8, 
    top_p=0.4, 
    max_new_tokens=20,
    num_beams=1,
    pad_token_id=None, 
    eos_token_id=None
)

# Print result data structureand confidence of each chain
print(output)
print("--------------------------------")

# Decode and print the left-most output chain up to depth=3 (new thought on each line)
decodedTree = multisampling.decodeTree(output, tokenizer)
print("\n Root node: \n")
print(decodedTree[0]["output"])
print("\n Depth = 1: \n")
print(decodedTree[1][0]["output"])
print("\n Depth = 2: \n")
print(decodedTree[1][1][0]["output"])
print("\n Depth = 3: \n")
print(decodedTree[1][1][1][0]["output"])

# Visualize tree using pyqt5
app = visualization.QApplication(visualization.sys.argv)
view = visualization.FlowchartView()
view.setWindowTitle("CoT Visualization")
view.show()

# Display nodes
visualization.createNodes(view,tokenizer,view.windowWidth/2,0,output,width=350,shiftAmt=600,dropAmt=350,shiftReduction=1.6)

# Exit application when window closed
visualization.sys.exit(app.exec_())
