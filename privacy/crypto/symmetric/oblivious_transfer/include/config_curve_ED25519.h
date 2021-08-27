#ifndef CONFIG_CURVE_ED25519_H
#define CONFIG_CURVE_ED25519_H

#include"amcl.h"
#include"config_field_25519.h"

// ECP stuff

#define CURVETYPE_ED25519 EDWARDS
#define PAIRING_FRIENDLY_ED25519 NOT
#define CURVE_SECURITY_ED25519 128


#if PAIRING_FRIENDLY_ED25519 != NOT
//#define USE_GLV_ED25519	  /**< Note this method is patented (GLV), so maybe you want to comment this out */
//#define USE_GS_G2_ED25519 /**< Well we didn't patent it :) But may be covered by GLV patent :( */
#define USE_GS_GT_ED25519 /**< Not patented, so probably safe to always use this */

#define POSITIVEX 0
#define NEGATIVEX 1

#define SEXTIC_TWIST_ED25519 
#define SIGN_OF_X_ED25519 

#define ATE_BITS_ED25519 

#endif

#if CURVE_SECURITY_ED25519 == 128
#define AESKEY_ED25519 16 /**< Symmetric Key size - 128 bits */
#define HASH_TYPE_ED25519 SHA256  /**< Hash type */
#endif

#if CURVE_SECURITY_ED25519 == 192
#define AESKEY_ED25519 24 /**< Symmetric Key size - 192 bits */
#define HASH_TYPE_ED25519 SHA384  /**< Hash type */
#endif

#if CURVE_SECURITY_ED25519 == 256
#define AESKEY_ED25519 32 /**< Symmetric Key size - 256 bits */
#define HASH_TYPE_ED25519 SHA512  /**< Hash type */
#endif



#endif
