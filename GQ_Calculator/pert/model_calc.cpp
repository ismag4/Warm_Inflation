
#include <boost/math/special_functions/lambert_w.hpp>
#include "Bg.cpp"

/* Set from the Python script find_GQ.py */

double kp = 0.05;

double EM_step = 1e-5;
int Nrealz = 2048;

int want_Np_autocalc = 0;
int verbose = 0;
int want_FP = 1;
int wi2easy = 0;
int unscaled = 0;

/* ###################################### */

extern "C" {

    void model(double phi_ini, double Q_ini, double gst,
               double V0, int p, int c, int therm,
               int rad_noise, int hybrid_inf) {

        double Cr = (M_PI * M_PI / 30.0) * gst;

        double php_ini = 0.0;
        double T_ini = 0.0;
        double Cy = 0.0;
        double Np = 0.0;


        // ============================================================
        // 1. Inflationary potential: Quartic model
        //
        //    V(phi) = V0*phi^4/4
        // ============================================================

        auto V = [V0](double phi) -> double {
            return (V0 / 4.0) * pow(phi, 4.0);
        };

        auto Vd = [V0](double phi) -> double {
            return V0 * pow(phi, 3.0);
        };

        auto Vdd = [V0](double phi) -> double {
            return 3.0 * V0 * pow(phi, 2.0);
        };


        // ============================================================
        // 2. Initial conditions
        // ============================================================

        auto set_php_ini = [Vd, V, phi_ini, &php_ini]
                           (double Q_ini) -> void {

            php_ini = -Vd(phi_ini) /
                      (V(phi_ini) * (1.0 + Q_ini));
        };

        set_php_ini(Q_ini);


        auto set_T_ini = [V, phi_ini, php_ini, Cr, &T_ini]
                         (double Q_ini) -> void {

            T_ini = pow(
                (Q_ini * V(phi_ini) * php_ini * php_ini) /
                (4.0 * Cr),
                1.0 / 4.0
            );
        };

        set_T_ini(Q_ini);


        // ============================================================
        // 3. QCD parameters
        // ============================================================

        const double Nc = 3.0;
        const double Nf = 6.0;
        const double Nf_tilde = 5.0;

        const double alpha_MZ = 0.1172;
        const double MZ_GeV = 91.188;
        const double Mpl_GeV = 2.44e18;


        // ============================================================
        // 4. Running strong coupling alpha_g(T)
        // ============================================================

        auto alpha_g = [Nc, Nf, alpha_MZ, MZ_GeV, Mpl_GeV]
                       (double T) -> double {

            double T_GeV = T * Mpl_GeV;

            double b = (11.0 * Nc - 2.0 * Nf) /
                       (6.0 * M_PI);

            double alpha_inv =
                1.0 / alpha_MZ +
                b * log(T_GeV / MZ_GeV);

            return 1.0 / alpha_inv;
        };


        // ============================================================
        // 5. Lambert W argument
        // ============================================================

        auto lambert_argument = [Nc, Nf, alpha_g]
                                (double T) -> double {

            double alpha = alpha_g(T);

            return (exp(3.041) / Nc) *
                   sqrt(
                       (2.0 * M_PI * (2.0 * Nc + Nf)) /
                       (3.0 * alpha)
                   );
        };


        // ============================================================
        // 6. Temperature-dependent kappa(T)
        // ============================================================

        auto kappa = [Nc, Nf, lambert_argument]
                     (double T) -> double {

            double x = lambert_argument(T);

            double W = boost::math::lambert_w0(x);

            double prefactor =
                1.26 * (Nc * Nc - 1.0) /
                ((2.0 * Nc + Nf) * Nc);

            return prefactor * W;
        };


        // ============================================================
        // 7. Sphaleron factor A(T)
        //
        //    Upsilon_sph = A(T)/f_a^2
        // ============================================================

        auto A_T = [Nc, alpha_g, kappa]
                   (double T) -> double {

            double alpha = alpha_g(T);
            double kap = kappa(T);

            return 0.5 * kap *
                   pow(Nc * alpha, 5.0) *
                   pow(T, 3.0);
        };


        // ============================================================
        // 8. Hubble parameter: slow-roll approximation
        //
        //    H_SR^2 = V(phi)/3
        // ============================================================

        auto H_slow = [V](double phi) -> double {

            return sqrt(V(phi) / 3.0);
        };


        // ============================================================
        // 9. Effective dissipation correction
        //
        //    B(phi,T) = 4*Nf_tilde*A(T) /
        //               (Nc*T^2*H_SR(phi))
        // ============================================================

        auto B_eff = [A_T, H_slow, Nc, Nf_tilde]
                     (double phi, double T) -> double {

            double H = H_slow(phi);

            return (4.0 * Nf_tilde * A_T(T)) /
                   (Nc * T * T * H);
        };


        // ============================================================
        // 10. Effective dissipation without Cy
        //
        //     Upsilon_wo_Cy = A(T)/(1+B)
        // ============================================================

        auto Ups_wo_Cy = [A_T, B_eff]
                         (double phi, double T) -> double {

            double B = B_eff(phi, T);

            return A_T(T) / (1.0 + B);
        };


        // ============================================================
        // 11. Determine Cy from initial Q
        // ============================================================

        double H_ini = H_slow(phi_ini);

        Cy = (3.0 * H_ini * Q_ini) /
             Ups_wo_Cy(phi_ini, T_ini);


        // ============================================================
        // 12. Full effective dissipation coefficient
        // ============================================================

        auto Ups = [Cy, Ups_wo_Cy]
                   (double phi, double T) -> double {

            return Cy * Ups_wo_Cy(phi, T);
        };


        // ============================================================
        // 13. Logarithmic temperature derivative of A(T)
        //
        //     d ln A/dT =
        //     [3 - 5*b*alpha + b*alpha/(2*(1+W))]/T
        // ============================================================

        auto dlogA_dT = [Nc, Nf, alpha_g, lambert_argument]
                        (double T) -> double {

            double alpha = alpha_g(T);

            double W = boost::math::lambert_w0(
                lambert_argument(T)
            );

            double b = (11.0 * Nc - 2.0 * Nf) /
                       (6.0 * M_PI);

            return (
                3.0 - 5.0 * b * alpha
                + (b * alpha) / (2.0 * (1.0 + W))
            ) / T;
        };


        // ============================================================
        // 14. Partial derivative with respect to temperature
        //
        //     d ln B/dT = d ln A/dT - 2/T
        // ============================================================

        auto pT_Ups = [Cy, Ups_wo_Cy, B_eff, dlogA_dT]
                      (double phi, double T) -> double {

            double B = B_eff(phi, T);

            double dlnA = dlogA_dT(T);

            double dlnB = dlnA - 2.0 / T;

            double U0 = Ups_wo_Cy(phi, T);

            return Cy * U0 *
                   (
                       dlnA -
                       (B / (1.0 + B)) * dlnB
                   );
        };


        // ============================================================
        // 15. Partial derivative with respect to inflaton field
        //
        //     d ln B/dphi = -V_phi/(2V)
        //
        //     d ln Upsilon/dphi =
        //     B/(1+B) * V_phi/(2V)
        // ============================================================

        auto pph_Ups = [Cy, Ups_wo_Cy, B_eff, V, Vd]
                       (double phi, double T) -> double {

            double B = B_eff(phi, T);

            double U0 = Ups_wo_Cy(phi, T);

            return Cy * U0 *
                   (B / (1.0 + B)) *
                   (Vd(phi) / (2.0 * V(phi)));
        };


        // ============================================================
        // 16. Solve perturbations and calculate G(Q)
        //
        //     GQ perturbation solver
        // ============================================================

        bg_solver(
            V,
            Vd,
            Vdd,
            Ups,
            pT_Ups,
            pph_Ups,
            Cr,
            Np,
            phi_ini,
            php_ini,
            T_ini,
            therm,
            kp,
            EM_step,
            Nrealz,
            want_Np_autocalc,
            verbose,
            rad_noise,
            hybrid_inf,
            want_FP,
            wi2easy,
            unscaled
        );

    }


    // ================================================================
    // 17. Clear global variables
    // ================================================================

    void clear_vars() {
        Nend = 0.0;
        GQ_val = 0.0;
        Q_val = 0.0;
    }


    void set_phi_crit(double x) {
        ph_crit = x;
    }


    // ================================================================
    // 18. Set global variables from Python
    // ================================================================

    void set_globals(
        int N_realizations,
        double Nstar,
        int verbosity,
        int FP_approach,
        int want_wi2easy,
        int want_unscaled
    ) {

        Nrealz = N_realizations;
        Nevol = Nstar;
        verbose = verbosity;
        want_FP = FP_approach;
        wi2easy = want_wi2easy;
        unscaled = want_unscaled;

    }

}
