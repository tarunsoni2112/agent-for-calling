import unittest

from negotiation_engine import (
    ContextValidationError,
    NegotiationContext,
    build_payload,
    build_user_message,
    load_model_settings,
)


class NegotiationEngineTests(unittest.TestCase):
    def _base_context(self) -> NegotiationContext:
        return NegotiationContext(
            name="Ravi",
            requirement="CRM automation",
            language_mode="Hinglish",
            min_price=12000,
            ideal_price=15000,
            walkaway_price=11000,
            summary="Customer comparing options.",
            latest_user_input="Aap best price batao.",
        )

    def test_load_model_settings_has_temperature(self) -> None:
        settings = load_model_settings()
        self.assertIn("temperature", settings)

    def test_build_user_message_replaces_placeholders(self) -> None:
        message = build_user_message(self._base_context())
        self.assertIn("Customer Name: Ravi", message)
        self.assertIn("ideal_price: 15000", message)
        self.assertNotIn("{{name}}", message)

    def test_validation_rejects_unsupported_language(self) -> None:
        context = self._base_context()
        context = NegotiationContext(**{**context.__dict__, "language_mode": "English"})

        with self.assertRaises(ContextValidationError):
            build_user_message(context)

    def test_validation_rejects_invalid_pricing_relationship(self) -> None:
        context = self._base_context()
        context = NegotiationContext(**{**context.__dict__, "ideal_price": 10000})

        with self.assertRaises(ContextValidationError):
            build_user_message(context)

    def test_build_payload_shape(self) -> None:
        payload = build_payload(self._base_context())
        self.assertIn("system_prompt", payload)
        self.assertIn("user_message", payload)
        self.assertIn("model_settings", payload)


if __name__ == "__main__":
    unittest.main()
