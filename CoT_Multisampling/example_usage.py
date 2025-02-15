import multisampling
import visualization
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Choose device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move to chosen device

# Encode input prompt
prompt = 'If you rearrange the letters in "new door," what two words do you get?'
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device) 

# List of nodal structure for each node in the tree
nodes = [3,1,1] # First node generation has 2 branches, second has 1 branch, third has 1 branch

# Generate output using multisampling and specific inference parameters
output = multisampling.CoTTreeTokens(
    model, input_ids, nodes, True, 
    temperature=0.9, 
    top_p=0.7, 
    max_new_tokens=40,
    num_beams=1, 
    pad_token_id=None, 
    eos_token_id=None
)

# Print result data structureand confidence of each chain
print(output)
print("--------------------------------")

# Decode and print the left-most output chain up to depth=2 (new thought on each line)
print(prompt)
print(tokenizer.decode(output[1][0]["output"], skip_special_tokens=True))
print(tokenizer.decode(output[1][1][0]["output"], skip_special_tokens=True))

# Visualize tree using pyqt5
app = visualization.QApplication(visualization.sys.argv)
view = visualization.FlowchartView()
view.setWindowTitle("CoT Visualization")
view.show()

# Display nodes
visualization.createNodes(view,tokenizer,view.windowWidth/2,0,output,width=350,shiftAmt=600,dropAmt=350,shiftReduction=1.6)

# Exit application when window closed
visualization.sys.exit(app.exec_())
