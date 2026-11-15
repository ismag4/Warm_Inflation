
import numpy as np
from cffi import FFI

ffi = FFI()


# ============================================================
# 1. Numerical settings
# ============================================================

kp = 0.05

em_step = 1e-5
Nrealz = 2048

kmax = 2.0
kmin = -6.0
points_k = 50

Np_autocalc = 1
verbosity = 1

# Deterministic perturbation solver
want_FP = 1

# WI2easy-compatible quantum prescription
wi2easy = 1


# ============================================================
# 2. Output settings
# ============================================================

write_bg = True

fname_bg = "bg.dat"
fname_ps = "ps.dat"


# ============================================================
# 3. Rudnei model parameters
# ============================================================

# Radiation degrees of freedom
# SM particles only: 106.75
# SM particles + thermalized inflaton: 107.75

therm = 0
gst = 106.75

# Inflationary parameters
# IMPORTANT: These values are provisional.
# They must be replaced by the parameters selected
# to reproduce the reference model.

V0 = 2.05237e-15

Q0 = 0.311553
phi0 = 23.6895

Np = 1.0

# Radiation perturbation noise
rad_noise = 0


# ============================================================
# 4. Legacy interface parameters
# ============================================================

# These parameters are retained only because the
# current C++ model() interface requires them.
#
# They are NOT used to define Rudnei's dissipation.

p = 3
c = 0


# ============================================================
# 5. Run
# ============================================================

if __name__ == "__main__":

    lib_pert = ffi.dlopen("../libmodel.so")

    ffi.cdef("""
        void model (
            double phi_ini,
            double gst,
            double Q_ini,
            double V0,
            double Np,
            int p,
            int c,
            int therm,
            int rad_noise
        );

        void set_globals (
            double kpivot,
            double Em_h,
            int N_realizations,
            double kmax,
            double kmin,
            int points_bw_k,
            int Np_calc,
            int verbosity,
            int FP_approach,
            int want_wi2easy
        );

        int get_npts();

        double* get_klist();
        double* get_Plist();

        void clear_P();
        void clear_k();

        void write_Bg(const char* fname);

        extern double PT_kp;
    """, override=True)


    # Configure solver
    lib_pert.set_globals(
        kp,
        em_step,
        Nrealz,
        kmax,
        kmin,
        points_k,
        Np_autocalc,
        verbosity,
        want_FP,
        wi2easy
    )


    # Run model
    lib_pert.model(
        phi0,
        gst,
        Q0,
        V0,
        Np,
        p,
        c,
        therm,
        rad_noise
    )


    # Save background
    if write_bg:
        lib_pert.write_Bg(fname_bg.encode("utf-8"))


    # Retrieve numerical spectrum
    npts = lib_pert.get_npts()

    Pptr = lib_pert.get_Plist()
    kptr = lib_pert.get_klist()

    Plist = ffi.unpack(Pptr, npts)
    klist = ffi.unpack(kptr, npts)


    # Tensor power spectrum
    PT = lib_pert.PT_kp

    with open("PT_kp.dat", "w") as f_pt:
        f_pt.write(str(PT))


    # Save scalar power spectrum
    with open(fname_ps, "w") as f_ps:

        f_ps.write("k (in Mpc^-1), P(k)\n")

        for k, P in zip(klist, Plist):
            f_ps.write(f"{k},{P}\n")
