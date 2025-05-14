import pickle, random
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


alice, bob = ["Alice says:"], ["Bob says:"]
females = ["Abigail says:", "Alice says:", "Amanda says:", "Amy says:", "Andrea says:", "Angela says:", "Anna says:", "Ashley says:", "Ava says:", "Brittany says:", "Brooke says:", "Caitlin says:", "Caroline says:", "Chelsea says:", "Danielle says:", "Diana says:", "Elizabeth says:", "Emily says:", "Emma says:", "Eva says:", "Grace says:", "Hailey says:", "Hannah says:", "Isabella says:", "Jacqueline says:", "Jessica says:", "Jasmine says:", "Jennifer says:", "Jenna says:", "Julia says:", "Kaitlyn says:", "Katherine says:", "Kayla says:", "Kimberly says:", "Lily says:", "Lauren says:", "Leah says:", "Layla says:", "Lily says:", "Linda says:", "Marissa says:", "Megan says:", "Mia says:", "Maria says:", "Mary says:", "Melissa says:", "Natalie says:", "Nicole says:", "Olivia says:", "Paige says:"]
males = ["James says:", "John says:", "Robert says:", "Michael says:", "William says:", "David says:", "Richard says:", "Charles says:", "Joseph says:", "Thomas says:", "Mark says:", "Donald says:", "Christopher says:", "Paul says:", "George says:", "Stephen says:", "James says:", "Edward says:", "Steven says:", "Kenneth says:", "Brian says:", "Kevin says:", "Matthew says:", "Gary says:", "Eric says:", "Stephen says:", "Andrew says:", "Anthony says:", "Daniel says:", "Jacob says:", "Jason says:", "Douglas says:", "Charles says:", "Barry says:", "John says:", "Henry says:", "Scott says:", "Patrick says:", "Alexander says:", "Robert says:", "Nicholas says:", "Will says:", "Caleb says:", "Benjamin says:", "Jacob says:", "Noah says:", "Gavin says:", "Samuel says:", "Grayson says:", "Theodore says:"]
animals = ["The gerbil says:", "The hamster says:", "The dog says:", "The rabbit says:", "The cat says:", "The tiger says:", "The lion says:", "The pig says:", "The cow says:", "The monkey says:", "The elephant says:", "The antelope says:", "The panda says:", "The zebra says:", "The fox says:", "The squirrel says:", "The chipmunk says:", "The otter says:", "The seal says:", "The walrus says:", "The orca says:", "The dolphin says:", "The tiger says:", "The cheetah says:", "The jaguar says:", "The leopard says:", "The snow leopard says:", "The panther says:", "The cougar says:", "The lynx says:", "The bobcat says:", "The wildcat says:", "The sabertooth tiger says:", "The lion says:", "The tiger says:", "The jaguar says:", "The leopard says:", "The snow leopard says:", "The panther says:", "The cougar says:", "The lynx says:", "The bobcat says:", "The wildcat says:", "The dog says:", "The cat says:", "The bear says:", "The boar says:", "The armadillo says:", "The beaver says:", "The chinchilla says:"]
insects = ["The ant says:", "The aphid says:", "The assassin bug says:", "The bee says:", "The beetle says:", "The butterfly says:", "The cicada says:", "The cockroach says:", "The crane fly says:", "The cricket says:", "The damselfly says:", "The dragonfly says:", "The earwig says:", "The firefly says:", "The flea says:", "The fly says:", "The fruit fly says:", "The gnat says:", "The grasshopper says:", "The hornet says:", "The housefly says:", "The june bug says:", "The katydid says:", "The ladybug says:", "The leaf hopper says:", "The louse says:", "The locust says:", "The mantisfly says:", "The mayfly says:", "The midge says:", "The mosquito says:", "The moth says:", "The pill bug says:", "The praying mantis says:", "The scarab beetle says:", "The shield bug says:", "The silverfish says:", "The stick insect says:", "The stink bug says:", "The termite says:", "The thrip says:", "The tick says:", "The velvet ant says:", "The walkingstick says:", "The wasp says:", "The water bug says:", "The water strider says:", "The weevil says:", "The whitefly says:", "The yellowjacket says:"]
odds = [f"Q{i}:" for i in range(1, 100, 2)]
evens = [f"Q{i}:" for i in range(2, 100, 2)]
mornings = [f"[{random.randint(1, 11)}:{random.randint(10, 59)}] True or False?" for _ in range(100)]
afternoons = [f"[{random.randint(13, 24)}:{random.randint(10, 59)}] True or False?" for _ in range(100)]
informal = ["Hi.", "Hey.", "Hello.", "Yo.", "Sup.", "What's up.", "Howdy.", "Hiya.", "Hey there.", "Wassup.", "Wazzup.", "'Sup.", "What's good.", "What's crackin'.", "How's it going.", "How's life.", "How's tricks.", "What's new.", "Long time no see.", "Look who it is!", "Yoooo.", "Hola.", "Ayo.", "Heyyy.", "'Ello.", "Hi hi.", "Oi.", "Yo yo yo.", "What's happenin'.", "Top of the mornin'.", "How's things.", "You alright.", "How you doin'.", "What's the word.", "What's goin' on.", "What it do.", "Howdy-do.", "Greetings.", "Peace.", "Yo fam.", "Heya.", "Ahoy.", "Alright mate.", "Cheers.", "Hello stranger.", "G'day.", "Salutations.", "Good to see ya.", "'Ey up.", "Whaddup."]
formal = ["Good morning.", "Good afternoon.", "Good evening.", "Hello.", "Greetings.", "How do you do.", "It's a pleasure to meet you.", "Nice to meet you.", "Pleased to meet you.", "Good day.", "I hope you're well.", "I trust you are doing well.", "I hope this message finds you well.", "How are you today.", "How have you been.", "Welcome.", "It's good to see you.", "How do you do, sir.", "How do you do, madam.", "Salutations.", "A pleasure to make your acquaintance.", "Good to see you again.", "I hope all is well.", "Wishing you a good day.", "I hope everything is going smoothly.", "I trust things are going well.", "I hope you had a pleasant day.", "Warm greetings.", "Respectful greetings.", "Honored to meet you.", "Delighted to make your acquaintance.", "A very good morning to you.", "A very good afternoon to you.", "A very good evening to you.", "It's been a while.", "I'm glad we could meet.", "I appreciate your time.", "Thank you for joining me.", "Welcome aboard.", "Welcome, and thank you for being here.", "It's a privilege to meet you.", "It's an honor to meet you.", "I look forward to working with you.", "Thank you for taking the time.", "I hope you've been keeping well.", "Please accept my warmest greetings.", "It's nice to connect with you.", "Wishing you a pleasant day ahead."]


prefixes = {
    "ab": (alice, bob),
    "animal": (animals, insects),
    "gender": (females, males),
    "odd_even": (odds, evens),
    "time": (mornings, afternoons),
    "greeting": (informal, formal)
}


def _load_mistral(model_name: str, lora_path: str = None, get_n_layers: bool = False) -> tuple[AutoModelForImageTextToText, AutoProcessor, int]:
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.chat_template = processor.chat_template

    if get_n_layers:
        try: n_layers = model.config.num_hidden_layers
        except: n_layers = model.config.text_config.num_hidden_layers

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, processor.tokenizer, n_layers
    else:
        return model, processor.tokenizer

def load_model_and_tokenizer(model_name: str, lora_path: str = None, get_n_layers: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:
    # check for mistral
    if "mistral" in model_name:
        return _load_mistral(model_name, lora_path, get_n_layers) 

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if get_n_layers:
        try: n_layers = model.config.num_hidden_layers
        except: n_layers = model.config.text_config.num_hidden_layers

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, tokenizer, n_layers
    else:
        return model, tokenizer