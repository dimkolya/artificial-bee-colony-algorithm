import numpy

MAXITER = 10000

def gen(bounds):
    dimension = len(bounds)
    x = list()
    for bound in bounds:
        x.append(numpy.random.uniform(bound[0], bound[1]))
    return x


def annealing(f, bounds, max_it=MAXITER):
    best_x = None
    for i in range(10):
        x = gen(bounds)
        temperature = 1
        for it in range(max_it):
            new_x = [0] * len(x)
            dxs = numpy.random.uniform(-1, 1, size=len(x))
            for i in range(len(x)):
                new_x[i] = x[i] + dxs[i] * max(temperature, 0.01)

            if f(new_x) > f(x) or numpy.exp((f(new_x) - f(x)) / temperature) > numpy.random.rand():
                x = new_x

            temperature *= 0.99
        if best_x == None or f(best_x) < f(x):
            best_x = x
    return x


def abc(f, bounds, number=50, max_it=MAXITER, limit=10):
    food_number = number // 2
    dimension = len(bounds)
    xs = [gen(bounds) for i in range(food_number)]

    best_x = None
    trial = [0] * food_number
    for it in range(max_it):
        for i in range(food_number):
            j = numpy.random.randint(dimension)
            k = numpy.random.randint(dimension)
            new_x = xs[i].copy()
            phi = numpy.random.uniform(-1, 1)
            new_x[j] = xs[i][j] + phi * (xs[i][j] - xs[k][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        fit_min = numpy.min(list(map(lambda x: f(x), xs))) - 0.001
        fit_sum = numpy.sum(list(map(lambda x: f(x) - fit_min, xs)))
        p = list(map(lambda x: (f(x) - fit_min) / fit_sum, xs))

        indices = [i for i in range(food_number)]
        for t in range(food_number):
            j = numpy.random.randint(dimension)
            i = numpy.random.choice(indices, p=p)
            k = i
            while k == i:
                k = numpy.random.randint(food_number)
            new_x = xs[i].copy()
            phi = numpy.random.uniform(-1, 1)
            new_x[j] = xs[i][j] + phi * (xs[i][j] - xs[k][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(food_number):
            if trial[i] > limit:
                if best_x == None or f(best_x) < f(xs[i]):
                    best_x = xs[i]
                x = gen(bounds)
                xs[i] = x
    for x in xs:
        if best_x == None or f(x) > f(best_x):
            best_x = x
    return best_x


def cabc(f, bounds, number=50, max_it=MAXITER, limit=10, best_number=10):
    food_number = number // 2
    dimension = len(bounds)
    xs = [gen(bounds) for i in range(food_number)]

    best_x = None
    trial = [0] * food_number
    for it in range(max_it):
        best_xs = sorted(xs, key=f, reverse=True)[:best_number]
        for i in range(food_number):
            j = numpy.random.randint(dimension)
            r1 = numpy.random.randint(best_number)
            r2 = numpy.random.randint(best_number)
            while r1 == r2:
                r2 = numpy.random.randint(best_number)
            new_x = xs[i].copy()
            phi = numpy.random.uniform(-1, 1)
            new_x[j] = best_xs[r1][j] + phi * (best_xs[r1][j] - best_xs[r2][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x.copy()
                trial[i] = 0
            else:
                trial[i] += 1

        fit_min = numpy.min(list(map(lambda x: f(x), xs))) - 0.001
        fit_sum = numpy.sum(list(map(lambda x: f(x) - fit_min, xs)))
        p = list(map(lambda x: (f(x) - fit_min) / fit_sum, xs))

        indices = [i for i in range(food_number)]
        for t in range(food_number):
            j = numpy.random.randint(dimension)
            i = numpy.random.choice(indices, p=p)
            k = i
            while k == i:
                k = numpy.random.randint(food_number)
            new_x = xs[i].copy()
            phi = numpy.random.uniform(-1, 1)
            new_x[j] = xs[i][j] + phi * (xs[i][j] - xs[k][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(food_number):
            if trial[i] > limit:
                if best_x == None or f(best_x) < f(xs[i]):
                    best_x = xs[i]
                x = gen(bounds)
                xs[i] = x
    for x in xs:
        if best_x == None or f(x) > f(best_x):
            best_x = x
    return best_x


def m2abc(f, bounds, number=50, max_it=MAXITER, limit=10):
    food_number = number // 2
    dimension = len(bounds)
    xs = [gen(bounds) for i in range(food_number)]

    best_x = None
    trial = [0] * food_number
    for it in range(max_it):
        x_best = max(xs, key=f)
        for i in range(food_number):
            j = numpy.random.randint(dimension)
            r1 = numpy.random.randint(food_number)
            r2 = numpy.random.randint(food_number)
            while r1 == r2:
                r2 = numpy.random.randint(food_number)
            new_x = xs[i].copy()
            phi1 = numpy.random.uniform(-1, 1)
            phi2 = numpy.random.uniform(-1, 1)
            new_x[j] = (xs[r1][j] + xs[r2][j]) / 2 + phi1 * (xs[r1][j] - xs[r1][j]) + phi2 * (x_best[j] - xs[r1][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        fit_min = numpy.min(list(map(lambda x: f(x), xs))) - 0.001
        fit_sum = numpy.sum(list(map(lambda x: f(x) - fit_min, xs)))
        p = list(map(lambda x: (f(x) - fit_min) / fit_sum, xs))

        indices = [i for i in range(food_number)]
        for t in range(food_number):
            j = numpy.random.randint(dimension)
            i = numpy.random.choice(indices, p=p)
            k = i
            while k == i:
                k = numpy.random.randint(food_number)
            new_x = xs[i].copy()
            phi = numpy.random.uniform(-1, 1)
            new_x[j] = xs[i][j] + phi * (xs[i][j] - xs[k][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(food_number):
            if trial[i] > limit:
                if best_x == None or f(best_x) < f(xs[i]):
                    best_x = xs[i]
                x = gen(bounds)
                xs[i] = x
    for x in xs:
        if best_x == None or f(x) > f(best_x):
            best_x = x
    return best_x


def tsaabc(f, bounds, number=50, max_it=MAXITER, limit=10, best_number=10, pf=0.5):
    food_number = number // 2
    dimension = len(bounds)
    xs = [gen(bounds) for i in range(food_number)]

    best_x = None
    trial = [0] * food_number
    for it in range(max_it):
        best_xs = sorted(xs, key=f, reverse=True)[:best_number]
        x_best = max(xs, key=f)
        for i in range(food_number):
            j = numpy.random.randint(dimension)
            new_x = xs[i].copy()
            if numpy.random.uniform(0, 1) < pf:
                r1 = numpy.random.randint(food_number)
                r2 = numpy.random.randint(food_number)
                while r1 == r2:
                    r2 = numpy.random.randint(food_number)
                phi1 = numpy.random.uniform(-1, 1)
                phi2 = numpy.random.uniform(-1, 1)
                new_x[j] = (xs[r1][j] + xs[r2][j]) / 2 + phi1 * (xs[r1][j] - xs[r1][j]) + phi2 * (x_best[j] - xs[r1][j])
            else:
                r1 = numpy.random.randint(best_number)
                r2 = numpy.random.randint(best_number)
                while r1 == r2:
                    r2 = numpy.random.randint(best_number)
                phi = numpy.random.uniform(-1, 1)
                new_x[j] = best_xs[r1][j] + phi * (best_xs[r1][j] - best_xs[r2][j])

            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        fit_min = numpy.min(list(map(lambda x: f(x), xs))) - 0.001
        fit_sum = numpy.sum(list(map(lambda x: f(x) - fit_min, xs)))
        p = list(map(lambda x: (f(x) - fit_min) / fit_sum, xs))

        indices = [i for i in range(food_number)]
        for t in range(food_number):
            j = numpy.random.randint(dimension)
            i = numpy.random.choice(indices, p=p)
            k = i
            while k == i:
                k = numpy.random.randint(food_number)
            new_x = xs[i].copy()
            phi = numpy.random.uniform(-1, 1)
            new_x[j] = xs[i][j] + phi * (xs[i][j] - xs[k][j])
            if f(new_x) > f(xs[i]):
                xs[i] = new_x
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(food_number):
            if trial[i] > limit:
                if best_x == None or f(best_x) < f(xs[i]):
                    best_x = xs[i]
                x = gen(bounds)
                xs[i] = x
    for x in xs:
        if best_x == None or f(x) > f(best_x):
            best_x = x
    return best_x
