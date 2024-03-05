In the project directory, you can run:

```sh
python app.py
```

The API will be accessible at [http://localhost:9000](http://localhost:9000)

You can test it with:

```bash
curl -X GET -H "Content-Type: application/json" -d '{
    "data": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}' http://localhost:9000/`
```

and

```bash
curl -X GET -H "Content-Type: application/json" -d '{
  "data": {
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    "perimeter_mean": 87.46,
    "area_mean": 566.3,
    "smoothness_mean": 0.09779,
    "compactness_mean": 0.08129,
    "concavity_mean": 0.06664,
    "concave points_mean": 0.04781,
    "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699,
    "texture_se": 0.7886,
    "perimeter_se": 2.058,
    "area_se": 23.56,
    "smoothness_se": 0.008462,
    "compactness_se": 0.0146,
    "concavity_se": 0.02387,
    "concave points_se": 0.01315,
    "symmetry_se": 0.0198,
    "fractal_dimension_se": 0.0023,
    "radius_worst": 15.11,
    "texture_worst": 19.26,
    "perimeter_worst": 99.7,
    "area_worst": 711.2,
    "smoothness_worst": 0.144,
    "compactness_worst": 0.1773,
    "concavity_worst": 0.239,
    "concave points_worst": 0.1288,
    "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259
  }
}' http://localhost:9000/
```