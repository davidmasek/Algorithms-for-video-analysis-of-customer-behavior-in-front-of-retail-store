import math
import sys
import timeit


def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form.
    source: https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py
    """

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60*60*24),("h", 60*60),("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    
    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.  
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = [u"s", u"ms",u'us',"ns"] # the save value   
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms",u'\xb5s',"ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]
        
    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return u"%.*g %s" % (precision, timespan * scaling[order], units[order])


class TimeitResult:
    """
    source: https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py (modified)
    """
    def __init__(self, loops, repeat, best, worst, all_runs, precision=3):
        self.loops = loops
        self.repeat = repeat
        self.best = best
        self.worst = worst
        self.all_runs = all_runs
        self._precision = precision
        self.timings = [ dt / self.loops for dt in all_runs]

    @property
    def average(self):
        return math.fsum(self.timings) / len(self.timings)

    @property
    def stdev(self):
        mean = self.average
        return (math.fsum([(x - mean) ** 2 for x in self.timings]) / len(self.timings)) ** 0.5

    def __str__(self):
        pm = '+-'
        # Try to use unicode pm characater
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            try:
                u'\xb1'.encode(sys.stdout.encoding)
                pm = u'\xb1'
            except:
                pass
        return (
            u"{mean} {pm} {std} per loop (mean {pm} std. dev. of {runs} run{run_plural}, {loops} loop{loop_plural} each)"
                .format(
                    pm = pm,
                    runs = self.repeat,
                    loops = self.loops,
                    loop_plural = "" if self.loops == 1 else "s",
                    run_plural = "" if self.repeat == 1 else "s",
                    mean = _format_time(self.average, self._precision),
                    std = _format_time(self.stdev, self._precision))
                )

    def _repr_pretty_(self, p , cycle):
        unic = self.__str__()
        p.text(u'<TimeitResult : '+unic+u'>')

def timethat(stmt, setup='pass', globals=None, loops=None, repeat=7, quiet=False):
    timer = timeit.Timer(stmt=stmt, setup=setup, globals=globals)

    if loops is None:
        loops, _ = timer.autorange()
    
    all_runs = timer.repeat(repeat, loops)
    best = min(all_runs) / loops
    worst = max(all_runs) / loops
    timeit_result = TimeitResult(loops, repeat, best, worst, all_runs)

    if not quiet :
        # Check best timing is greater than zero to avoid a
        # ZeroDivisionError.
        # In cases where the slowest timing is lesser than a microsecond
        # we assume that it does not really matter if the fastest
        # timing is 4 times faster than the slowest timing or not.
        if worst > 4 * best and best > 0 and worst > 1e-6:
            print("The slowest run took %0.2f times longer than the "
                    "fastest. This could mean that an intermediate result "
                    "is being cached." % (worst / best))

    return timeit_result
