
#include <boost/math/special_functions/lambert_w.hpp>
#include "Bg.cpp"

double kp = 0.05; // Set from Python script
int verbose = 0;  // Set from Python script

extern "C" {

    void model(double phi_ini, double Q_ini, double gst, double V0,
               int p, int c, int hybrid_inf) {

        double Cr = (M_PI * M_PI / 30.0) * gst;
        double php_ini = 0.0;
        double T_ini = 0.0;
        double Cy = 0.0;

        // ============================================================
        // 1. Inflationary potential: Quartic model
        //    V(phi) = V0 * phi^4 / 4
        // ============================================================

        auto V = [V0](double phi) -> double {
            return (V0 / 4.0) * pow(phi, 4.0);
        };

        auto Vd = [V0](double phi) -> double {
            return V0 * pow(phi, 3.0);
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
        // 7. Sphaleron temperature-dependent factor A(T)
        //
        //    Upsilon_sph = A(T) / f_a^2
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
        //    H_SR^2 = V(phi) / 3
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
        // 10. Effective dissipation without normalization Cy
        //
        //     Upsilon_wo_Cy = A(T) / (1 + B)
        // ============================================================

        auto Ups_wo_Cy = [A_T, B_eff]
                         (double phi, double T) -> double {

            double B = B_eff(phi, T);

            return A_T(T) / (1.0 + B);
        };


        // ============================================================
        // 11. Determine Cy from the initial dissipation ratio
        //
        //     Q_ini = Upsilon_ini / (3*H_ini)
        //
        //     Cy = 3*H_ini*Q_ini / Upsilon_wo_Cy_ini
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
        // 13. Solve background evolution
        //
        //     GQ background solver interface
        // ============================================================

        bg_solver(
            V,
            Vd,
            Ups,
            Cr,
            phi_ini,
            php_ini,
            T_ini,
            kp,
            verbose,
            hybrid_inf
        );

    }


    // ================================================================
    // 14. Auxiliary functions required by SWIM
    // ================================================================

    void clear_Nend() {
        Nend = 0.0;
    }


    void set_phi_crit(double x) {
        ph_crit = x;
    }

}
