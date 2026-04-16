"""
openrouter_wrapper.py — TextAttack ModelWrapper for OpenRouter API models.

Wraps a cloud LLM (GPT-4o-mini, Gemma, etc.) as a black-box text classifier
so TextAttack can attack it without needing local weights or gradients.

Supports only black-box attacks (TextFooler, PWWS, BAE, etc.).
Gradient-based attacks (HotFlip, BERTAttack) are NOT compatible.
"""

import time
import re
import numpy as np
import requests
from textattack.models.wrappers import ModelWrapper


class OpenRouterWrapper(ModelWrapper):
    """
    Black-box ModelWrapper that sends text to an OpenRouter-hosted LLM
    and returns a one-hot probability vector over the label set.

    TextAttack's __call__ contract:
        Input : list of strings
        Output: np.ndarray of shape [len(text_list), num_classes]
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        label2id: dict,
        id2label: dict,
        prompt_template: str,
        call_delay: float = 1.5,
        max_retries: int = 5,
        retry_base_wait: int = 10,
    ):
        """
        Args:
            model_id        : OpenRouter model string, e.g. "openai/gpt-4o-mini"
            api_key         : OpenRouter API key
            base_url        : OpenRouter base URL (https://openrouter.ai/api/v1)
            label2id        : dict mapping label string → int index
            id2label        : dict mapping int index → label string
            prompt_template : prompt with {text} placeholder for the input sentence
            call_delay      : seconds to wait between API calls (rate-limit buffer)
            max_retries     : max retries on transient errors / rate limits
            retry_base_wait : base seconds for exponential backoff
        """
        self.model_id        = model_id
        self.api_key         = api_key
        self.base_url        = base_url.rstrip("/")
        self.label2id        = label2id
        self.id2label        = id2label
        self.prompt_template = prompt_template
        self.call_delay      = call_delay
        self.max_retries     = max_retries
        self.retry_base_wait = retry_base_wait
        self.total_queries   = 0
        self._cache: dict    = {}   # text → predicted label index

    # ------------------------------------------------------------------
    # TextAttack interface
    # ------------------------------------------------------------------

    def __call__(self, text_input_list):
        """Return shape [n, num_classes] probability array."""
        outputs = []
        for text in text_input_list:
            outputs.append(self._classify(str(text)))
        return np.array(outputs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """Return predicted string label for a single text (uses cache)."""
        probs = self._classify(text)
        return self.id2label[int(np.argmax(probs))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, text: str) -> np.ndarray:
        if text in self._cache:
            idx = self._cache[text]
        else:
            prompt   = self.prompt_template.format(text=text)
            response = self._call_api(prompt)
            label    = self._parse_label(response)
            idx      = self.label2id.get(label, 0)
            self._cache[text] = idx

        probs      = np.zeros(len(self.label2id))
        probs[idx] = 1.0
        return probs

    def _call_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/thesis-adversarial-nlp",
            "X-Title":       "Adversarial NLP Attack",
        }
        payload = {
            "model":       self.model_id,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens":  20,   # only need a short label, not a long response
        }

        resp = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.call_delay)
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                self.total_queries += 1
                return resp.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.HTTPError:
                code = resp.status_code if resp is not None else 0
                if code in (429, 503):
                    wait = self.retry_base_wait * (2 ** attempt)
                    print(f"  [RATE LIMIT {code}] Waiting {wait}s "
                          f"(attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(wait)
                elif code == 402:
                    raise RuntimeError(
                        "Insufficient OpenRouter credits. "
                        "Add credits at https://openrouter.ai/settings/credits"
                    )
                else:
                    raise
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_wait)
                else:
                    raise
        return ""

    def _parse_label(self, response: str) -> str:
        """Map a free-text model response back to a known label string."""
        response = response.strip()

        # Exact match (case-insensitive)
        for label in self.label2id:
            if label.lower() == response.lower():
                return label

        # Substring match
        for label in self.label2id:
            if label.lower() in response.lower():
                return label

        print(f"  [WARN] Could not parse label from response: {repr(response[:80])}")
        return response
