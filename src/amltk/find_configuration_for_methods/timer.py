import timeit
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

f = open("times.txt", "a")

original_time = timeit.timeit(stmt='get_dataset(16)', setup='from src.amltk.datasets.Datasets import get_dataset', number=1)
print(original_time)
f.write("Original Time: " + str(original_time) + "\n")

autofeat_time = timeit.timeit(stmt='get_autofeat_features(train_x, train_y, test_x, task_hint, 1)', setup='from src.amltk.datasets.Datasets import get_dataset; from src.amltk.feature_engineering.Autofeat import get_autofeat_features; train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=16)', number=1)
print(autofeat_time)
f.write("Autofeat Time: " + str(autofeat_time) + "\n")


openfe_time = timeit.timeit(stmt='get_openFE_features(train_x, train_y, test_x, 1)', setup='from src.amltk.datasets.Datasets import get_dataset; from src.amltk.feature_engineering.OpenFE import get_openFE_features; train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=16)', number=1)
print(openfe_time)
f.write("OpenFE Time: " + str(openfe_time) + "\n")


autogluon_time = timeit.timeit(stmt='get_autogluon_features(train_x, train_y, test_x)', setup='from src.amltk.datasets.Datasets import get_dataset; from src.amltk.feature_engineering.AutoGluon import get_autogluon_features; train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=16)', number=1)
print(autogluon_time)
f.write("AutoGluon Time: " + str(autogluon_time) + "\n")

mljar_time = timeit.timeit(stmt='get_mljar_features(train_x, train_y, test_x, 1000)', setup='from src.amltk.datasets.Datasets import get_dataset; from src.amltk.feature_engineering.MLJAR import get_mljar_features; train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=16)', number=1)
print(mljar_time)
f.write("MLJar Time: " + str(mljar_time) + "\n")

