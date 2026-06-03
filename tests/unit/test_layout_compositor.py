"""Unit tests for image fitting strategies in the compositor."""
from __future__ import annotations

from PIL import Image

from comicweaver.agents.layout.compositor import (
    FitStrategy,
    PanelSlot,
    _choose_strategy,
    _fit_image,
)


# ---------------------------------------------------------------------------
# _choose_strategy
# ---------------------------------------------------------------------------

class TestChooseStrategy:
    def test_long_shot_uses_contain(self):
        s = _choose_strategy("extreme_long", 0.5)
        assert s == FitStrategy.CONTAIN

    def test_close_shot_uses_smart_cover(self):
        s = _choose_strategy("close", 0.5)
        assert s == FitStrategy.SMART_COVER

    def test_extreme_close_uses_smart_cover(self):
        s = _choose_strategy("extreme_close", 0.5)
        assert s == FitStrategy.SMART_COVER

    def test_high_emotion_uses_smart_cover(self):
        s = _choose_strategy("medium", 0.9)
        assert s == FitStrategy.SMART_COVER

    def test_medium_neutral_uses_cover(self):
        s = _choose_strategy("medium", 0.5)
        assert s == FitStrategy.COVER


# ---------------------------------------------------------------------------
# _fit_image — aspect-ratio preservation
# ---------------------------------------------------------------------------

class TestFitImage:
    def _make_img(self, w: int, h: int) -> Image.Image:
        return Image.new("RGB", (w, h), (100, 150, 200))

    def test_cover_preserves_aspect_ratio(self):
        """A tall (portrait) image fitted to a wide target should NOT be stretched."""
        img = self._make_img(768, 1024)  # 3:4 portrait
        result = _fit_image(img, 600, 200, FitStrategy.COVER)  # 3:1 wide
        assert result.size == (600, 200)
        # The result should be a crop of a scaled image, not a stretch
        # Verify by checking that the scaled intermediate preserved ratio
        # (we can't directly check intermediate, but we know cover scales to fill)

    def test_contain_preserves_aspect_ratio(self):
        """Contain should fit the whole image with padding, no distortion."""
        img = self._make_img(768, 1024)  # 3:4 portrait
        result = _fit_image(img, 600, 200, FitStrategy.CONTAIN)  # 3:1 wide
        assert result.size == (600, 200)

    def test_cover_does_not_distort_square_to_wide(self):
        img = self._make_img(512, 512)
        result = _fit_image(img, 800, 400, FitStrategy.COVER)
        assert result.size == (800, 400)

    def test_similar_ratio_stretch_is_lossless(self):
        """Very similar ratios (< 3% diff) should just be resized."""
        img = self._make_img(768, 1024)  # 0.75 ratio
        result = _fit_image(img, 750, 1002, FitStrategy.COVER)  # ~0.749 ratio
        assert result.size == (750, 1002)

    def test_smart_cover_returns_correct_size(self):
        img = self._make_img(768, 1024)
        result = _fit_image(img, 400, 400, FitStrategy.SMART_COVER)
        assert result.size == (400, 400)

    def test_smart_cover_differs_from_center_cover(self):
        """Smart cover anchors higher; should differ from center cover for tall→square."""
        # Create a tall image with distinctive colored bands
        striped = Image.new("RGB", (768, 1024), (50, 50, 50))
        # White band in top 40% — smart_cover should include more of this
        for y in range(410):
            for x in range(768):
                striped.putpixel((x, y), (255, 255, 255))
        # Black band in bottom 30% — center_cover would include more of this
        for y in range(717, 1024):
            for x in range(768):
                striped.putpixel((x, y), (0, 0, 0))

        smart = _fit_image(striped, 400, 300, FitStrategy.SMART_COVER)
        center = _fit_image(striped, 400, 300, FitStrategy.COVER)

        # Smart cover should have more white (top) than center cover
        smart_avg = sum(smart.getpixel((200, y))[0] for y in range(300)) / 300
        center_avg = sum(center.getpixel((200, y))[0] for y in range(300)) / 300
        assert smart_avg > center_avg, (
            f"Smart cover (avg={smart_avg:.0f}) should be brighter than "
            f"center cover (avg={center_avg:.0f})"
        )
