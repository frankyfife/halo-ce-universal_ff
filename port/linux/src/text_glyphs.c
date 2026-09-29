/*
TEXT_GLYPHS.C

Sharp text at the display's resolution.

The game draws text from a 128x128 cache texture (rasterizer_text.c): each
glyph a font needs is copied there from the font's 8-bit coverage, as 4-bit
alpha, and drawn with bilinear filtering at the 480-line layout's size. On a
large display that magnifies 4-bit glyphs several times, which blurs their
edges into steps, and the filter reads the neighbouring glyphs at each
quad's edges.

The game reports every glyph it caches (halo_text_glyph_cached), with the
font's 8-bit coverage. This file keeps a copy of the cache that is scale
times larger, where scale is the screen's pixels per layout unit, rounded
up. Each glyph is magnified alone (Catmull-Rom, outside the glyph is empty),
and the coverage is then sharpened around its middle so that an edge takes
about one output pixel: magnified coverage is close to a distance to the
edge, as in distance field text. A stroke fainter than full coverage keeps
its strength: the threshold and the result scale with the strongest
coverage the filter reads (all 4x4 texels, so that the soft rim of a full
stroke is not taken for a faint stroke of its own). The device binds this copy wherever the game
binds the cache texture (d3d8_gl.c bind_textures); texture coordinates of
the cache are texel coordinates scaled by its size, so the larger copy
needs no change to them.

display.sharp_text = false draws the game's own cache.
*/

#include "xgpu.h"
#include "port_config.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_UNPACK_ROW_LENGTH
#define GL_UNPACK_ROW_LENGTH 0x0cf2
#endif

#define MAXIMUM_TEXT_SCALE 8
#define MAXIMUM_GLYPH_RECORDS 256

struct glyph_record
{
	short x0, y0, width, height;
};

static struct
{
	BOOL attached;
	BOOL disabled;
	const void *game_texture;       /* the game's cache: its Direct3D texture */
	long width, height;             /* of the game's cache */
	float sharpness;                /* display.text_sharpness, 0 to 1 */
	unsigned char *coverage;        /* width x height, 8-bit */
	struct glyph_record glyphs[MAXIMUM_GLYPH_RECORDS];
	unsigned long glyph_next;

	long scale;                     /* of the copy; 0 before the first bind */
	unsigned char *pixels;          /* RGBA, (width * scale) x (height * scale) */
	GLuint texture;
	/* the part of the copy that changed since the last upload, in cache texels */
	long dirty_x0, dirty_y0, dirty_x1, dirty_y1;
} text;

static void dirty_add(long x0, long y0, long x1, long y1)
{
	if (text.dirty_x1 <= text.dirty_x0 || text.dirty_y1 <= text.dirty_y0)
	{
		text.dirty_x0 = x0;
		text.dirty_y0 = y0;
		text.dirty_x1 = x1;
		text.dirty_y1 = y1;
		return;
	}
	if (x0 < text.dirty_x0) text.dirty_x0 = x0;
	if (y0 < text.dirty_y0) text.dirty_y0 = y0;
	if (x1 > text.dirty_x1) text.dirty_x1 = x1;
	if (y1 > text.dirty_y1) text.dirty_y1 = y1;
}

/* ---------- magnification */

static float glyph_texel(const struct glyph_record *glyph, long x, long y)
{
	if (x < 0 || y < 0 || x >= glyph->width || y >= glyph->height)
		return 0.0f;
	return text.coverage[(glyph->y0 + y) * text.width + glyph->x0 + x] * (1.0f / 255.0f);
}

static void catmull_rom_weights(float t, float *weights)
{
	float t2 = t * t, t3 = t2 * t;

	weights[0] = 0.5f * (-t3 + 2.0f * t2 - t);
	weights[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
	weights[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
	weights[3] = 0.5f * (t3 - t2);
}

static BOOL glyphs_overlap(const struct glyph_record *a, const struct glyph_record *b)
{
	return a->x0 < b->x0 + b->width && b->x0 < a->x0 + a->width &&
		a->y0 < b->y0 + b->height && b->y0 < a->y0 + a->height;
}

/* whether a strip of the cache is free of glyphs the game may still draw,
other than glyph */
static BOOL strip_unused(const struct glyph_record *strip, const struct glyph_record *glyph)
{
	unsigned long index;

	for (index = 0; index < MAXIMUM_GLYPH_RECORDS; index++)
	{
		const struct glyph_record *other = &text.glyphs[index];

		if (other != glyph && other->width && glyphs_overlap(other, strip))
			return FALSE;
	}
	return TRUE;
}

/* writes a glyph's texels of the copy, and clears the texel row below and
the column right of it where they hold what an older glyph left there: the
filter reads half a texel past the glyph's quad */
static void glyph_magnify(const struct glyph_record *glyph)
{
	long scale = text.scale, pitch = text.width * scale;
	long output_width = glyph->width * scale, output_height = glyph->height * scale;
	long ox, oy;

	for (oy = 0; oy < output_height; oy++)
	{
		float sy = ((float)oy + 0.5f) / (float)scale - 0.5f;
		long iy = (long)floorf(sy);
		float wy[4];

		catmull_rom_weights(sy - (float)iy, wy);
		for (ox = 0; ox < output_width; ox++)
		{
			float sx = ((float)ox + 0.5f) / (float)scale - 0.5f;
			long ix = (long)floorf(sx);
			float wx[4], value = 0.0f, strongest = 0.0f, alpha;
			int i, j;
			unsigned char *pixel;

			catmull_rom_weights(sx - (float)ix, wx);
			for (j = 0; j < 4; j++)
			{
				float row = 0.0f;

				for (i = 0; i < 4; i++)
				{
					float texel = glyph_texel(glyph, ix - 1 + i, iy - 1 + j);

					row += wx[i] * texel;
					if (texel > strongest)
						strongest = texel;
				}
				value += wy[j] * row;
			}
			if (strongest < 1.0f / 255.0f)
			{
				alpha = 0.0f;
			}
			else
			{
				float soft = value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

				alpha = (value - 0.5f * strongest) * (float)scale / strongest + 0.5f;
				alpha = alpha < 0.0f ? 0.0f : alpha > 1.0f ? 1.0f : alpha;
				alpha *= strongest;
				/* display.text_sharpness: from the smooth magnification (0) to
				the sharpened edge (1) */
				alpha = soft + (alpha - soft) * text.sharpness;
			}
			/* a quad's edge can fall inside a screen pixel, where the filter
			reads the neighbouring glyph of the cache: an empty rim a fraction
			of a glyph texel wide keeps the neighbours apart */
			if (scale >= 3 && (ox == 0 || oy == 0 || ox == output_width - 1 || oy == output_height - 1))
				alpha = 0.0f;
			pixel = text.pixels + ((glyph->y0 * scale + oy) * pitch + glyph->x0 * scale + ox) * 4;
			pixel[0] = pixel[1] = pixel[2] = 255;
			pixel[3] = (unsigned char)(alpha * 255.0f + 0.5f);		}
	}
	{
		struct glyph_record below = { glyph->x0, (short)(glyph->y0 + glyph->height), glyph->width, 1 };
		struct glyph_record right = { (short)(glyph->x0 + glyph->width), glyph->y0, 1, glyph->height };

		if (below.y0 < text.height && strip_unused(&below, glyph))
		{
			unsigned char *row = text.pixels + ((below.y0 * scale) * pitch + glyph->x0 * scale) * 4;

			for (ox = 0; ox < output_width; ox++)
				row[ox * 4 + 3] = 0;
		}
		if (right.x0 < text.width && strip_unused(&right, glyph))
		{
			for (oy = 0; oy < output_height; oy++)
				text.pixels[((glyph->y0 * scale + oy) * pitch + right.x0 * scale) * 4 + 3] = 0;
		}
	}
	dirty_add(glyph->x0, glyph->y0, glyph->x0 + glyph->width + 1, glyph->y0 + glyph->height + 1);
}

/* ---------- the game's side */

void halo_text_glyph_cached(const void *cache_texture, long cache_width, long cache_height,
	short x0, short y0, short width, short height, const unsigned char *coverage)
{
	struct glyph_record *glyph;
	unsigned long index;
	long y;

	if (text.disabled)
		return;
	if (!text.attached)
	{
		text.disabled = !config_boolean("display.sharp_text");
		if (text.disabled)
			return;
		text.game_texture = cache_texture;
		text.sharpness = (float)config_real("display.text_sharpness");
		text.sharpness = text.sharpness < 0.0f ? 0.0f : text.sharpness > 1.0f ? 1.0f : text.sharpness;
		text.width = cache_width;
		text.height = cache_height;
		text.coverage = calloc((size_t)(cache_width * cache_height), 1);
		text.attached = text.coverage != NULL;
		text.disabled = !text.attached;
		if (text.disabled)
			return;
	}
	if (width <= 0 || height <= 0 || x0 < 0 || y0 < 0 || x0 + width > text.width || y0 + height > text.height)
		return;
	for (y = 0; y < height; y++)
		memcpy(text.coverage + (y0 + y) * text.width + x0, coverage + y * width, (size_t)width);

	/* the new glyph replaces the ones it covers */
	glyph = &text.glyphs[text.glyph_next];
	glyph->x0 = x0;
	glyph->y0 = y0;
	glyph->width = width;
	glyph->height = height;
	for (index = 0; index < MAXIMUM_GLYPH_RECORDS; index++)
	{
		if (index != text.glyph_next && text.glyphs[index].width && glyphs_overlap(&text.glyphs[index], glyph))
			text.glyphs[index].width = 0;
	}
	text.glyph_next = (text.glyph_next + 1) % MAXIMUM_GLYPH_RECORDS;
	if (text.pixels)
		glyph_magnify(glyph);
}

/* ---------- the device's side */

GLuint xgpu_text_glyphs_texture(const void *texture, float screen_scale)
{
	long scale;

	if (!text.attached || texture != text.game_texture)
		return 0;
	scale = (long)ceilf(screen_scale - 0.01f);
	if (scale < 1)
		scale = 1;
	if (scale > MAXIMUM_TEXT_SCALE)
		scale = MAXIMUM_TEXT_SCALE;
	if (scale != text.scale)
	{
		unsigned long index;
		unsigned char *pixels = calloc((size_t)(text.width * scale * text.height * scale), 4);

		if (!pixels)
			return 0;
		free(text.pixels);
		text.pixels = pixels;
		text.scale = scale;
		for (index = 0; index < MAXIMUM_GLYPH_RECORDS; index++)
		{
			if (text.glyphs[index].width)
				glyph_magnify(&text.glyphs[index]);
		}
		if (!text.texture)
			glGenTextures(1, &text.texture);
		glBindTexture(GL_TEXTURE_2D, text.texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)(text.width * scale), (GLsizei)(text.height * scale), 0,
			GL_RGBA, GL_UNSIGNED_BYTE, text.pixels);
		xgpu_gl_state_invalidate();
		text.dirty_x0 = text.dirty_x1 = 0;
		platform_log("text: glyphs drawn %ld times larger", scale);
	}
	if (text.dirty_x1 > text.dirty_x0 && text.dirty_y1 > text.dirty_y0)
	{
		long x0 = text.dirty_x0 * scale, y0 = text.dirty_y0 * scale;
		long x1 = (text.dirty_x1 < text.width ? text.dirty_x1 : text.width) * scale;
		long y1 = (text.dirty_y1 < text.height ? text.dirty_y1 : text.height) * scale;

		glBindTexture(GL_TEXTURE_2D, text.texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)(text.width * scale));
		glTexSubImage2D(GL_TEXTURE_2D, 0, (GLint)x0, (GLint)y0, (GLsizei)(x1 - x0), (GLsizei)(y1 - y0), GL_RGBA,
			GL_UNSIGNED_BYTE, text.pixels + (y0 * text.width * scale + x0) * 4);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		xgpu_gl_state_invalidate();
		text.dirty_x0 = text.dirty_x1 = 0;
	}
	return text.texture;
}
