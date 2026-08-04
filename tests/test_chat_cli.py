import unittest

from scripts.chat_cli import generate, generation_kwargs


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return "generated"


class FakeTokenizer:
    eos_token_id = 42


class ChatCliGenerationTests(unittest.TestCase):
    def test_generation_settings_are_loaded_from_config(self):
        settings = generation_kwargs(
            {
                "max_new_tokens": 128,
                "do_sample": True,
                "temperature": 0.65,
                "top_p": 0.82,
                "repetition_penalty": 1.12,
                "no_repeat_ngram_size": 3,
            },
            pad_token_id=42,
        )
        self.assertEqual(settings["temperature"], 0.65)
        self.assertEqual(settings["top_p"], 0.82)
        self.assertEqual(settings["repetition_penalty"], 1.12)
        self.assertEqual(settings["no_repeat_ngram_size"], 3)

    def test_generate_passes_no_repeat_ngram_size(self):
        model = FakeModel()
        result = generate(
            model,
            {"input_ids": [1, 2, 3]},
            FakeTokenizer(),
            {"no_repeat_ngram_size": 3},
        )
        self.assertEqual(result, "generated")
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 3)
        self.assertEqual(model.kwargs["pad_token_id"], 42)


if __name__ == "__main__":
    unittest.main()
