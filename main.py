import interegular

from transformers import AutoTokenizer
from interegular.fsm import anything_else

pattern = '\{[ ]?"color"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?\}'
fsm = interegular.parse_pattern(pattern=pattern).to_fsm().reduce()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
vocabulary = {}
special_tokens = set(tokenizer.all_special_tokens)

# initial -> state
initial = fsm.initial

# masks: Dict[state, Sequence[token_id]]
masks = {}

# token_fsm: Dict[Tuple[state, token], state]
token_fsm = {}

for token, token_idx in tokenizer.get_vocab().items():
    if token not in special_tokens:
        vocabulary[token] = token_idx

        for init_state in fsm.states:

            is_valid = True

            state = init_state

            for symbol in token:

                if anything_else in fsm.alphabet and not symbol in fsm.alphabet:
                    symbol = anything_else

                transition = fsm.alphabet[symbol]

                if not (state in fsm.map and transition in fsm.map[state]):
                    is_valid = False
                    break

                state = fsm.map[state][transition]

            if is_valid:
                token_fsm[(init_state, token_idx)] = state
                masks.setdefault(init_state, []).append(token_idx)


print("Summary:")
print(f"Number of transitions: {len(token_fsm)}")
print(f"Number of states:{len(masks)}")

