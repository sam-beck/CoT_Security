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

# Number of nodes in each chain
nodes = 2

# Generate output using multisampling and specific inference parameters
output = multisampling.CoTTreeTokens(
    model, input_ids, 2, True, 
    temperature=0.9, 
    top_p=0.7, 
    num_return_sequences=nodes,
    max_new_tokens=40,
    num_beams=1, 
    pad_token_id=None, 
    eos_token_id=None
)



# Print result and confidence of each chain
print(output)
print("--------------------------------")

# Decode and print the left-most output chain up to depth=2 (new thought on each line)
print(prompt)
print(tokenizer.decode(output[1][0]["output"], skip_special_tokens=True))
print(tokenizer.decode(output[1][1][0]["output"], skip_special_tokens=True))

app = visualization.QApplication(visualization.sys.argv)
view = visualization.FlowchartView()
view.setWindowTitle("CoT Visualization")
view.show()

# Display nodes
visualization.createNodes(view,tokenizer,view.windowWidth/2,0,output,width=350,shiftAmt=600,dropAmt=350,shiftReduction=1.6)

visualization.sys.exit(app.exec_())
