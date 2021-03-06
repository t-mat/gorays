package javarays;

import static javarays.Camera.EYE_OFFSET;
import static javarays.Camera.ORIGIN;
import static javarays.Camera.RIGHT;
import static javarays.Camera.UP;
import static javarays.Camera.aspectRatio;

import java.util.concurrent.ThreadLocalRandom;

final class Worker implements Runnable  {

    // Default pixel color is almost pitch black
    private final RayVector DEFAULT_COLOR    = new RayVector(13, 13, 13);

    private static final RayVector EMPTY_VEC = new RayVector();
    private static final RayVector SKY_VEC   = new RayVector(1.f,  1.f,  1.f);

    private static final RayVector FLOOR_PATTERN_1 = new RayVector( 3.f,  1.f,  1.f);
    private static final RayVector FLOOR_PATTERN_2 = new RayVector( 3.f,  3.f,  3.f);

    private static final RayVector STD_VEC   = new RayVector(0.f,  0.f,  1.f);

    private final int offset;
    private final int jump;

    // for stochastic sampling
    private final ThreadLocalRandom rnd = ThreadLocalRandom.current();

    private final RayVector[] objects;
    private final RayImage image;

    private float t;
    private RayVector n;

    public Worker(final RayImage _image, final RayVector[] _objects, final int _offset, final int _jump) {
        image = _image;
        objects = _objects;
        offset = _offset;
        jump = _jump;
    }

    //The intersection test for line [orig, v].
    // Return 2 if a hit was found (and also return distance t and bouncing ray n).
    // Return 0 if no hit was found but ray goes upward
    // Return 1 if no hit was found but ray goes downward
    private final int tracer(final RayVector orig, final RayVector dir) {
        t = 1e9f;
        int m = 0;
        final float p = -orig.z / dir.z;

        n = EMPTY_VEC;
        if (.01f < p) {
            t = p;
            n = STD_VEC;
            m = 1;
        }

        for(int i = 0; i < objects.length; i++) {
            // There is a sphere but does the ray hits it ?
            final RayVector p1 = orig.add(objects[i]);
            final float b = p1.dot(dir);
            final float c = p1.dot(p1) - 1;
            final float b2 = b * b;

            // Does the ray hit the sphere ?
            if (b2 > c) {
                // It does, compute the distance camera-sphere
                final float q = b2 - c;
                final float s = (float) (-b - Math.sqrt(q));

                if (s < t && s > .01f) {
                    t = s;
                    n = (p1.add(dir.scale(t))).norm();
                    m = 2;
                }
            }
        }

        return m;
    }

    // Sample the world and return the pixel color for
    // a ray passing by point origin and dir (Direction)
    private final RayVector sample(final RayVector origin, final RayVector dir) {
        // Search for an intersection ray Vs World.
        final int m = tracer(origin, dir);
        final RayVector on = new RayVector(n);

        if (m == 0) { // m==0
            // No sphere found and the ray goes upward: Generate a sky color
            final float p = 1 - dir.z;
            return SKY_VEC.scale(p);
        }

        // A sphere was maybe hit.
        RayVector h = origin.add(dir.scale(t)); // h = intersection coordinate

        // 'l' = direction to light (with random delta for soft-shadows).
        final RayVector l = new RayVector(9.f + rnd.nextFloat(),
                                          9.f + rnd.nextFloat(),
                                          16.f).add(h.scale(-1.f)).norm();

        // Calculated the lambertian factor
        float b = l.dot(n);

        // Calculate illumination factor (lambertian coefficient > 0 or in shadow)?
        if (b < 0 || tracer(h, l) != 0) {
            b = 0;
        }

        if (m == 1) { // m == 1
            h = h.scale(.2f); // No sphere was hit and the ray was going downward: Generate a floor color
            final boolean cond = ((int) (Math.ceil(h.x) + Math.ceil(h.y)) & 1) == 1;
            return (cond ? FLOOR_PATTERN_1 : FLOOR_PATTERN_2).scale(b * .2f + .1f);
        }

        final RayVector r = dir.add(on.scale(on.dot(dir.scale(-2.f)))); // r = The half-vector

        // Calculate the color 'p' with diffuse and specular component
        final float p = (float)Math.pow(l.dot(r.scale(b > 0 ? 1.f : 0.f)), 99.0);

        // m == 2 A sphere was hit. Cast an ray bouncing from the sphere surface.
        // Attenuate color by 50% since it is bouncing (*.5)
        return new RayVector(p, p, p).add(sample(h, r).scale(.5f));
    }

    @Override
    public void run() {
        for (int y = offset; y < image.size; y += jump) { // For each row
            int k = (image.size - y - 1) * image.size * 3;

            for (int x = image.size; x-- > 0 ; ) { // For each pixel in a line
                // Reuse the vector class to store not XYZ but a RGB pixel color
                final RayVector p = innerLoop(y, x, DEFAULT_COLOR);
                image.data[k++] = clamp(p.x);
                image.data[k++] = clamp(p.y);
                image.data[k++] = clamp(p.z);
            }
        }
    }

    private RayVector innerLoop(final int y, final int x, RayVector p) {
        // Cast 64 rays per pixel (For blur (stochastic sampling)
        // and soft-shadows.
        for (int r = 0; r < 64; r++) {
            // The delta to apply to the origin of the view (For
            // Depth of View blur).
            final float factor1 = (rnd.nextFloat()-.5f) * 99.f;
            final float factor2 = (rnd.nextFloat()-.5f) * 99.f;
            final RayVector t = UP.scale(factor1).add(RIGHT.scale(factor2)); // A little bit of delta up/down and left/right

            // Set the camera focal point vector(17,16,8) and Cast the ray
            // Accumulate the color returned in the p variable

            // Ray Direction with random deltas
            final RayVector tmpA = UP.scale(rnd.nextFloat() + x * aspectRatio);
            final RayVector tmpB = RIGHT.scale(rnd.nextFloat() + y * aspectRatio);
            final RayVector tmpC = tmpA.add(tmpB).add(EYE_OFFSET);
            final RayVector rayDirection = t.scale(-1).add(tmpC.scale(16.f)).norm();

            p = sample(ORIGIN.add(t), rayDirection).scale(3.5f).add(p); // +p for color accumulation
        }
        return p;
    }

    private final byte clamp(final float v) {
        if (v > 255.f) {
            return (byte) 255;
        }
        return (byte) v;
    }
}
