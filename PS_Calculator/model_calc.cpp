#include <boost/math/special_functions/lambert_w.hpp> //Lambert Function
#include "Bg.cpp"

/* Set from the python scripts */
double kp = 0.05;
//int therm : Thermalization of inflaton, 1 = yes = Bose-Einstein; 0 = no

double EM_step = 1e-5; //SDE solver step-size after Q>1e4 
int Nrealz = 2500; //Number of realizations for SDE

//Klist lower and upper bounds in log10
double klow = -6.0;
double kup = 2.0;
int npts = 40; 

int want_Np_autocalc = 1; //Set to 1 if you want the solver to calculate Np itself otherwise set 0. In both cases supply Np
int verbose = 0; //Set to one if you want to see the error messages
int want_FP = 1; //enable the Fokker-Planck (deterministic) approach
int wi2easy = 0; //enable (1) or disable (0) wi2easy like treatment of quantum-noise
/* ######################### */

extern "C" {

    void model (double phi_ini,double gst,double Q_ini,double V0,double Np,int p,int c,int therm,int rad_noise) {

        double Cr =  ((M_PI*M_PI) / 30.0) * gst;
        double php_ini=0.0;
        double T_ini=0.0;
        double Cy=0.0;

        //#### Model Definition here ####//

        
        //#### Quartic model definition based in arXiv 2504.20943 ####//

        // ============================================================
        // 1. Inflationary potential
        // ============================================================

        // ============================================================

	auto V = [V0](double phi) -> double {
	return (V0/4.0)*pow(phi,4.0);	
	};

	auto Vd = [V0](double phi) -> double {
    	return V0*pow(phi,3.0);
	};

	auto Vdd = [V0](double phi) -> double {
    	return 3.0*V0*pow(phi,2.0);
	};


// ============================================================
// 2. Initial conditions
// ============================================================

	auto set_php_ini = [Vd,V,phi_ini,&php_ini]
                   (double Q_ini) -> void {
    	php_ini = -Vd(phi_ini)/(V(phi_ini)*(1.0+Q_ini));
	};

	set_php_ini(Q_ini);


	auto set_T_ini = [V,phi_ini,php_ini,Cr,&T_ini]
                 (double Q_ini) -> void {
    	T_ini = pow(
        	(Q_ini*V(phi_ini)*php_ini*php_ini)/(4.0*Cr),
        	1.0/4.0
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

	auto alpha_g = [Nc,Nf,alpha_MZ,MZ_GeV,Mpl_GeV]
               (double T) -> double {

    	double T_GeV = T*Mpl_GeV;

    	double b = (11.0*Nc-2.0*Nf)/(6.0*M_PI);

    	double alpha_inv =
        	1.0/alpha_MZ + b*log(T_GeV/MZ_GeV);

    	return 1.0/alpha_inv;
	};


// ============================================================
// 5. Temperature-dependent kappa(T)
// ============================================================

	auto lambert_argument = [Nc,Nf,alpha_g]
                        (double T) -> double {

    	double alpha = alpha_g(T);

    	return (exp(3.041)/Nc)*
           	sqrt(
               	(2.0*M_PI*(2.0*Nc+Nf))/(3.0*alpha)
           	);
	};


	auto kappa = [Nc,Nf,lambert_argument]
             (double T) -> double {

    	double x = lambert_argument(T);

    	double W = boost::math::lambert_w0(x);

    	double prefactor =
        	1.26*(Nc*Nc-1.0)/
        	((2.0*Nc+Nf)*Nc);

    	return prefactor*W;
	};


// ============================================================
// 6. Sphaleron temperature-dependent factor A(T)
//
// Upsilon_sph = A(T)/f_a^2
// ============================================================

	auto A_T = [Nc,alpha_g,kappa]
           	(double T) -> double {

    	double alpha = alpha_g(T);
    	double kap = kappa(T);

    	return 0.5*kap*pow(Nc*alpha,5.0)*pow(T,3.0);
	};


// ============================================================
// 7. Hubble parameter: Rudnei slow-roll approximation
//
// H_SR^2 = V(phi)/3
//
// IMPORTANT:
// This approximation is used ONLY inside Upsilon_eff.
// SWIM continues to calculate the full Hubble parameter.
// ============================================================

	auto H_slow = [V](double phi) -> double {

    	return sqrt(V(phi)/3.0);
	};


// ============================================================
// 8. Effective dissipation correction
//
// B(phi,T) = 4*Nf_tilde*A(T)/(Nc*T^2*H_SR(phi))
// ============================================================

	auto B_eff = [A_T,H_slow,Nc,Nf_tilde]
        	     (double phi,double T) -> double {

    	double H = H_slow(phi);

    	return (4.0*Nf_tilde*A_T(T))/
           	(Nc*T*T*H);
	};


// ============================================================
// 9. Effective dissipation without C_Upsilon
//
// Upsilon_eff = A(T)/(1+B)
// ============================================================

	auto Ups_wo_Cy = [A_T,B_eff]
                 (double phi,double T) -> double {

    	double B = B_eff(phi,T);

    	return A_T(T)/(1.0+B);
	};


// ============================================================
// 10. Determine C_Upsilon from initial Q
//
// Q_ini = Upsilon_ini/(3*H_ini)
//
// Therefore:
// C_Upsilon = 3*H_ini*Q_ini/Upsilon_wo_Cy_ini
// ============================================================

	double H_ini = H_slow(phi_ini);

	Cy = (3.0*H_ini*Q_ini)/
     	Ups_wo_Cy(phi_ini,T_ini);


// ============================================================
// 11. Full effective dissipation coefficient
// ============================================================

	auto Ups = [Cy,Ups_wo_Cy]
           	(double phi,double T) -> double {

    	return Cy*Ups_wo_Cy(phi,T);
	};


// ============================================================
// 12. Derivatives of the effective dissipation
// ============================================================

// ------------------------------------------------------------
// 12.1 Logarithmic temperature derivative of A(T)
//
// d ln A / dT =
// [3 - 5*b*alpha + b*alpha/(2*(1+W))]/T
// ------------------------------------------------------------

	auto dlogA_dT = [Nc,Nf,alpha_g,lambert_argument]
                (double T) -> double {

    	double alpha = alpha_g(T);

    	double W = boost::math::lambert_w0(
        	lambert_argument(T)
    	);

    	double b = (11.0*Nc-2.0*Nf)/
        	       (6.0*M_PI);

    	return (
        	3.0 - 5.0*b*alpha
        	+ (b*alpha)/(2.0*(1.0+W))
    	)/T;
	};


// ------------------------------------------------------------
// 12.2 Partial derivative with respect to temperature
//
// d ln B/dT = d ln A/dT - 2/T
// ------------------------------------------------------------

	auto pT_Ups = [Cy,Ups_wo_Cy,B_eff,dlogA_dT]
              	(double phi,double T) -> double {

    	double B = B_eff(phi,T);

    	double dlnA = dlogA_dT(T);

    	double dlnB = dlnA - 2.0/T;

    	double U0 = Ups_wo_Cy(phi,T);

    	return Cy*U0*
           	(
               	dlnA -
               	(B/(1.0+B))*dlnB
           	);
	};


// ------------------------------------------------------------
// 12.3 Partial derivative with respect to inflaton field
//
// d ln B/dphi = -V_phi/(2V)
//
// Therefore:
// d ln Upsilon_eff/dphi =
// [B/(1+B)]*V_phi/(2V)
// ------------------------------------------------------------

	auto pph_Ups = [Cy,Ups_wo_Cy,B_eff,V,Vd]
               	(double phi,double T) -> double {

    	double B = B_eff(phi,T);

    	double U0 = Ups_wo_Cy(phi,T);

    	return Cy*U0*
           	(B/(1.0+B))*
           	(Vd(phi)/(2.0*V(phi)));
	};


// ============================================================
// End quartic model definition
// ============================================================



        /*
        //Find a value of Q_ini from phi_ini
        //DO NOT USE THIS FUNCTION IF YOUR UPSILON CAN NOT BE WRITTEN IN THE FORM: T^p * \phi^c . SUPPLY A Q_INITIAL INSTEAD
        auto Qi_find = [V,Vd,Cr,p,c,Cy,phi_ini] (double Qi) -> double {
            Qi = pow(10.0,Qi);
            return ( pow((1.0+Qi),(2.0*p))*pow(Qi,(4.0-p)) ) - ( (pow(Cy,4.0)/(9.0*pow(4.0,p)*pow(Cr,p))) * pow(phi_ini,(4.0*c)) * ( pow(Vd(phi_ini),(2.0*p))/pow(V(phi_ini),(p+2.0))  ) ) ;
        };

        try {
            auto res = boost::math::tools::bisect(Qi_find, -20.0, 5.0,root_stop(),max_iter);
            Q_ini = (res.second + res.first)/2;
            Q_ini = pow(10.0,Q_ini);
        }
        catch (const exception& e) {
            if (verbose==1) {
                cout<<"Q_initial couldn't be found"<<endl;
            }
            return;
        }
        */
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        bg_solver (V,Vd,Vdd,Ups,pT_Ups,pph_Ups,Cr,Np,phi_ini,php_ini,T_ini,therm,kp,klow,kup,EM_step,npts,Nrealz,want_Np_autocalc,want_FP,wi2easy,verbose,rad_noise); //Calculates the power-spectrum
    }  //Model Specification ends

    //Function to set the global variables
    void set_globals (double kpivot, double Em_h, int N_realizations, double kmax, double kmin, int points_bw_k, int Np_calc, int verbosity, int FP_approach,int want_wi2easy) {
        kp = kpivot;
        EM_step = Em_h;
        Nrealz = N_realizations;
        klow = kmin;
        kup = kmax;
        npts = points_bw_k;
        want_Np_autocalc = Np_calc;
        verbose = verbosity;
        want_FP = FP_approach;
        wi2easy = want_wi2easy;
    }

    // Define a function that returns a pointer to the global array
    int get_npts () {
        int npts = 0;
        try {
            npts = Plist.size();
        }
        catch (const exception& e) {
            npts = 0;
        }
        return npts;
    }

    double* get_Plist() {
        double* pt = Plist.data();
        return pt;
    }

    double* get_klist() {
        double* pt = klist.data();
        return pt;
    }
    
    void write_Bg(const char* fname) {
       ofstream fl(fname); //Stores in the format: N,phi,phi',T
       //Output header
       fl<<"N (e-folds), phi (in Mpl), phi' (in Mpl), T (in Mpl)"<<endl;
       for (int i=0;i<N_glob_array.size();i++) {
           fl<<N_glob_array[i]<<","<<phi_glob_array[i]<<","<<php_glob_array[i]<<","<<T_glob_array[i]<<endl;
       }
       fl.close();
    }
    
    void clear_P() {
        Plist = vector<double> ();
        Nend = 0.0;
        rngen.seed(std::chrono::system_clock::now().time_since_epoch().count() ^std::random_device{}() ^omp_get_thread_num() ^getpid()); //Reset global RNG after exit
    }
    void clear_k() { //Also clears bg vectors
        klist = vector<double> ();
        N_glob_array = vector<double> ();
        phi_glob_array = vector<double> ();
        php_glob_array = vector<double> ();
        T_glob_array = vector<double> ();
    }
}

