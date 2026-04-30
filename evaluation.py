import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# NUMBERS Task
num_before = load_json("metrics_Numbers_LSTM.json")
num_after = load_json("metrics_Numbers_MultiHead.json")

print("=" * 80)
print("NUMBERS - BEFORE MODEL (CNN-LSTM)")
print("=" * 80)
print(f"avg_loss                : {num_before['avg_loss']:.4f}")
print(f"approx_token_prob       : {num_before['approx_token_prob']:.2%}")
print(f"token_accuracy          : {num_before['token_accuracy']:.2%}")

print("\n" + "=" * 80)
print("NUMBERS - AFTER MODEL (MultiHead-CNN)")
print("=" * 80)
print(f"Dataset                        : {num_after['dataset']}")
print(f"Model                          : {num_after['model']}")
print(f"Best Val Exact Accuracy        : {num_after['best_val_exact_structured_accuracy']:.2%}")
print(f"Test Exact Sentence Accuracy   : {num_after['test_exact_structured_accuracy']:.2%}")

head_acc_num = num_after["test_head_accuracy"]
avg_token_num = sum(head_acc_num.values()) / len(head_acc_num)
print(f"Average Test Head Accuracy     : {avg_token_num:.2%}")

print("\nPer Attribute Head Accuracy:")
for key, acc in head_acc_num.items():
    print(f"   {key:15s}: {acc:.2%}")

last_epoch_num = num_after["history"][-1]
print("\nFinal Epoch Metrics:")
print(f"   Final Train Loss      : {last_epoch_num['train_loss']:.4f}")
print(f"   Final Val Loss        : {last_epoch_num['val_loss']:.4f}")
print(f"   Final Val Exact Acc   : {last_epoch_num['val_exact_structured_accuracy']:.2%}")
print("=" * 80)

# SHAPES Task
shape_lstm = load_json("metrics_Shapes_LSTM.json")
shape_multi = load_json("metrics_Shapes_MultiHead.json")

print("\n" + "=" * 80)
print("SHAPES - BEFORE MODEL (CNN-LSTM)")
print("=" * 80)
print(f"Dataset          : {shape_lstm['dataset']}")
print(f"Model            : {shape_lstm['model']}")
print(f"Test Loss        : {shape_lstm['test_loss']:.4f}")
print(f"Test Token Acc   : {shape_lstm['test_token_accuracy']:.2%}")
print(f"Test Sentence Acc: {shape_lstm['test_sentence_accuracy']:.2%}")
print(f"Best Val Token Acc: {shape_lstm['best_val_token_accuracy']:.2%}")

last_epoch_shape = shape_lstm["history"][-1]
print(f"Final Train Loss : {last_epoch_shape['train_loss']:.4f}")
print(f"Final Val Loss   : {last_epoch_shape['val_loss']:.4f}")
print(f"Final Val Token Acc: {last_epoch_shape['val_token_accuracy']:.2%}")
print(f"Final Val Sentence Acc: {last_epoch_shape['val_sentence_accuracy']:.2%}")

print("\n" + "=" * 80)
print("SHAPES - AFTER MODEL (MultiHead-CNN)")
print("=" * 80)
print(f"Dataset              : {shape_multi['dataset']}")
print(f"Model                : {shape_multi['model']}")
print(f"Best Val Struct Acc  : {shape_multi['best_val_exact_structured_accuracy']:.2%}")
print(f"Test Struct Acc      : {shape_multi['test_exact_structured_accuracy']:.2%}")

heads_shape = shape_multi["test_head_accuracy"]
avg_token_shape = sum(heads_shape.values()) / len(heads_shape)
print(f"Average Token Acc    : {avg_token_shape:.2%}")

print("\nTest Head Accuracy:")
for k, v in heads_shape.items():
    print(f"   {k:15s}: {v:.2%}")

last_epoch_multi = shape_multi["history"][-1]
print(f"\nFinal Train Loss      : {last_epoch_multi['train_loss']:.4f}")
print(f"Final Val Loss        : {last_epoch_multi['val_loss']:.4f}")
print(f"Final Val Struct Acc  : {last_epoch_multi['val_exact_structured_accuracy']:.2%}")
print("=" * 80)

# TIC TAC TOE Task
ttt_before = load_json("metrics_TicTacToe_LSTM.json")
ttt_after = load_json("metrics_TicTacToe_MultiHead.json")

print("\n" + "=" * 80)
print("TIC TAC TOE - BEFORE MODEL (CNN-LSTM)")
print("=" * 80)
print(f"avg_loss                : {ttt_before['avg_loss']:.4f}")
print(f"approx_token_prob       : {ttt_before['approx_token_prob']:.2%}")
print(f"token_accuracy          : {ttt_before['token_accuracy']:.2%}")

print("\n" + "=" * 80)
print("TIC TAC TOE - AFTER MODEL (MultiHead-CNN)")
print("=" * 80)
print(f"Dataset                        : {ttt_after['dataset']}")
print(f"Model                          : {ttt_after['model']}")
print(f"Best Val Exact Accuracy        : {ttt_after['best_val_exact_structured_accuracy']:.2%}")
print(f"Test Exact Sentence Accuracy   : {ttt_after['test_exact_structured_accuracy']:.2%}")

head_acc_ttt = ttt_after["test_head_accuracy"]
avg_token_ttt = sum(head_acc_ttt.values()) / len(head_acc_ttt)
print(f"Average Test Head Accuracy     : {avg_token_ttt:.2%}")

print("\nPer Position Head Accuracy:")
for pos, acc in head_acc_ttt.items():
    print(f"   {pos:15s}: {acc:.2%}")

last_epoch_ttt = ttt_after["history"][-1]
print("\nFinal Epoch Metrics:")
print(f"   Final Train Loss      : {last_epoch_ttt['train_loss']:.4f}")
print(f"   Final Val Loss        : {last_epoch_ttt['val_loss']:.4f}")
print(f"   Final Val Exact Acc   : {last_epoch_ttt['val_exact_structured_accuracy']:.2%}")

print("=" * 80)