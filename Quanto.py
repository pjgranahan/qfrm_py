from OptionValuation import OptionValuation, PriceSpec, Stock
from American import American
from math import sqrt
from numpy import cumsum, maximum, where, polyval, polyfit, exp, mean, copy
from numpy.random import normal, seed


class Quanto(OptionValuation):
    """ Quanto option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, vol_ex=0.0, correlation=0.0, seed=1,deg=5):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Calculates the value of a plain vanilla Quanto option.

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.
        seed: int
                MC random seed
        deg: int
                Degrees in LSM MC method.
        Returns
        -------
        self : Quanto

        .. sectionauthor:: Patrick Granahan, Runmin Zhang

        Notes
        -----

        Examples
        -------

        Calculate the price of a Quanto option. This example comes from Hull ch.30, ex.30.5 (p.701-702)

        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        >>> o.calc_px(method='LT', nsteps=100, vol_ex=0.12, correlation=0.2, keep_hist=True).px_spec.px
        179.82607364328157

        >>> o.px_spec.ref_tree # doctest: +ELLIPSIS
        ((1199.999999999993,), (1158.3148318698472, 1243.1853243866492), ... 38364.96926881886, 41175.99589789209))

        >>> o.calc_px(method='LT', nsteps=100, keep_hist=False)
        Quanto
        K: 1200
        T: 2
        _right: call
        _signCP: 1
        frf_r: 0.05
        px_spec: PriceSpec
          LT_specs:
            a: 1.0003000450045003
            d: 0.965262359891545
            df_T: 0.9048374180359595
            df_dt: 0.999000499833375
            dt: 0.02
            p: 0.49540447909174495
            u: 1.0359877703222138
          keep_hist: false
          method: LT
          nsteps: 100
          px: 172.20505562521683
          sub_method: binomial tree; Hull Ch.13
        ref: Stock
          S0: 1200
          curr: -
          desc: -
          q: 0.015
          tkr: -
          vol: 0.25
        rf_r: 0.03
        seed0: -
        <BLANKLINE>

        Calculate the price of a Quanto option. This example comes from Hull ch.30, problem.30.9.b (p.704)

        >>> s = Stock(S0=400, vol=.2, q=0.03)
        >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        >>> o.calc_px(method='LT', nsteps=100, vol_ex=0.06, correlation=0.4).px_spec.px
        57.50700503047851

        Example of option price development (LT method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='LT', nsteps=100, vol_ex=0.12, correlation=0.2).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        Calculate the price of a Quanto option using MC method. This example comes from Hull ch.30, ex.30.5 (p.701-702)
        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        >>> print(o.calc_px(method='MC', nsteps=100, npaths=5000,vol_ex=0.12, correlation=0.2).px_spec.px)
        179.88546563590577

        Calculate the price of a Quanto option. This example comes from Hull ch.30, problem.30.9.b (p.704)
        >>> s = Stock(S0=400, vol=.2, q=0.03)
        >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        >>> o.calc_px(method='MC', nsteps=100,npaths=4000, vol_ex=0.06, correlation=0.4).px_spec.px
        57.363490258590126

        Example of option price convergence (MC method) with increasing paths
        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='MC', nsteps=100, npaths=5000, vol_ex=0.12, correlation=0.2).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='MC Method: Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist,
                                 vol_ex=vol_ex, correlation=correlation,seed=1,deg=deg)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor:: Patrick Granahan

        """

        # Get provided parameters
        vol_ex = getattr(self.px_spec, 'vol_ex')  # Volatility of the exchange rate
        correlation = getattr(self.px_spec, 'correlation')  # Correlation between asset and exchange rate
        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)

        # Compute the foreign numeraire dividend yield
        growth_rate_of_underlying = (correlation * self.ref.vol * vol_ex)
        domestic_numeraire = self.rf_r - self.ref.q
        foreign_numeraire = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = self.frf_r - foreign_numeraire

        # Once we have the foreign numeraire dividend yield calculated,
        # we can price the Quanto option using an American option with specific parameters
        stock = Stock(S0=self.ref.S0, vol=self.ref.vol, q=foreign_numeraire_dividend_yield)
        american_option = American(ref=stock, right=self.right, K=self.K, T=self.T, rf_r=self.frf_r)

        # Then we take the price spec from the American option
        self.px_spec = american_option.calc_px(method='LT', nsteps=n, keep_hist=keep_hist).px_spec

        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor::

        Note
        ----


        """





        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor:: Runmin Zhang

        Note
        ----
        [1] http://unicreditanduniversities.eu/uploads/assets/QuantoAdjustments_in_the_Presence_of_SV_Giese.pdf
        """


        # Verify the input
        try: deg = self.px_spec.deg
        except TypeError: deg = 5

        vol_ex = getattr(self.px_spec, 'vol_ex')  # Volatility of the exchange rate
        correlation = getattr(self.px_spec, 'correlation')  # Correlation between asset and exchange rate
        n_steps = getattr(self.px_spec, 'nsteps', 3) # # of steps
        n_paths = getattr(self.px_spec, 'npaths', 5000) # of paths in MC simulation
        _ = self


        # Compute the foreign numeraire dividend yield
        growth_rate_of_underlying = (correlation * self.ref.vol * vol_ex)
        domestic_numeraire = self.rf_r - self.ref.q
        foreign_numeraire = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = self.frf_r - foreign_numeraire

        # Once we have the foreign numeraire dividend yield calculated,
        # Follow the LT method. We can price the Quanto option using an American option with specific parameters.

        dt = _.T / n_steps; df = exp(-_.frf_r * dt)
        signCP = 1 if _.right.lower()[0] == 'c' else -1

        seed(_.px_spec.seed)
        S = _.ref.S0 * exp(cumsum(normal((_.frf_r-foreign_numeraire_dividend_yield- 0.5 * _.ref.vol ** 2) * dt, _.ref.vol * sqrt(dt), (n_steps + 1, n_paths)), axis=0)); S[0] = _.ref.S0
        payout = maximum(signCP * (S - _.K), 0); v = copy(payout)  # terminal payouts

        for i in range(n_steps - 1, 0, -1):    # American Option Valuation by Backwards Induction
            rg = polyfit(S[i], v[i + 1] * df, deg)      # fit 5th degree polynomial to PV of current inner values
            C = polyval(rg, S[i])              # continuation values.
            v[i] = where(payout[i] > C, payout[i], v[i + 1] * df)  # exercise decision
        v[0] = v[1] * df
        self.px_spec.add(px=float(mean(v[0])))

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor::

        Note
        ----

        """

        return self

