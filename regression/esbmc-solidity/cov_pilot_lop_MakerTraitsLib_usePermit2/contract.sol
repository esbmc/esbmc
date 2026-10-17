// Sources flattened with hardhat v2.23.0 https://hardhat.org

// SPDX-License-Identifier: MIT

// File contracts/libraries/MakerTraitsLib.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.0;

type MakerTraits is uint256;

/**
 * @title MakerTraitsLib
 * @notice A library to manage and check MakerTraits, which are used to encode the maker's preferences for an order in a single uint256.
 * @dev
 * The MakerTraits type is a uint256 and different parts of the number are used to encode different traits.
 * High bits are used for flags
 * 255 bit `NO_PARTIAL_FILLS_FLAG`          - if set, the order does not allow partial fills
 * 254 bit `ALLOW_MULTIPLE_FILLS_FLAG`      - if set, the order permits multiple fills
 * 253 bit                                  - unused
 * 252 bit `PRE_INTERACTION_CALL_FLAG`      - if set, the order requires pre-interaction call
 * 251 bit `POST_INTERACTION_CALL_FLAG`     - if set, the order requires post-interaction call
 * 250 bit `NEED_CHECK_EPOCH_MANAGER_FLAG`  - if set, the order requires to check the epoch manager
 * 249 bit `HAS_EXTENSION_FLAG`             - if set, the order has extension(s)
 * 248 bit `USE_PERMIT2_FLAG`               - if set, the order uses permit2
 * 247 bit `UNWRAP_WETH_FLAG`               - if set, the order requires to unwrap WETH

 * Low 200 bits are used for allowed sender, expiration, nonceOrEpoch, and series
 * uint80 last 10 bytes of allowed sender address (0 if any)
 * uint40 expiration timestamp (0 if none)
 * uint40 nonce or epoch
 * uint40 series
 */
library MakerTraitsLib {
    // Low 200 bits are used for allowed sender, expiration, nonceOrEpoch, and series
    uint256 private constant _ALLOWED_SENDER_MASK = type(uint80).max;
    uint256 private constant _EXPIRATION_OFFSET = 80;
    uint256 private constant _EXPIRATION_MASK = type(uint40).max;
    uint256 private constant _NONCE_OR_EPOCH_OFFSET = 120;
    uint256 private constant _NONCE_OR_EPOCH_MASK = type(uint40).max;
    uint256 private constant _SERIES_OFFSET = 160;
    uint256 private constant _SERIES_MASK = type(uint40).max;

    uint256 private constant _NO_PARTIAL_FILLS_FLAG = 1 << 255;
    uint256 private constant _ALLOW_MULTIPLE_FILLS_FLAG = 1 << 254;
    uint256 private constant _PRE_INTERACTION_CALL_FLAG = 1 << 252;
    uint256 private constant _POST_INTERACTION_CALL_FLAG = 1 << 251;
    uint256 private constant _NEED_CHECK_EPOCH_MANAGER_FLAG = 1 << 250;
    uint256 private constant _HAS_EXTENSION_FLAG = 1 << 249;
    uint256 private constant _USE_PERMIT2_FLAG = 1 << 248;
    uint256 private constant _UNWRAP_WETH_FLAG = 1 << 247;

    /**
     * @notice Checks if the order has the extension flag set.
     * @dev If the `HAS_EXTENSION_FLAG` is set in the makerTraits, then the protocol expects that the order has extension(s).
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the flag is set.
     */
    function hasExtension(MakerTraits makerTraits) internal pure returns (bool) {
        return (MakerTraits.unwrap(makerTraits) & _HAS_EXTENSION_FLAG) != 0;
    }

    /**
     * @notice Checks if the maker allows a specific taker to fill the order.
     * @param makerTraits The traits of the maker.
     * @param sender The address of the taker to be checked.
     * @return result A boolean indicating whether the taker is allowed.
     */
    function isAllowedSender(MakerTraits makerTraits, address sender) internal pure returns (bool) {
        uint160 allowedSender = uint160(MakerTraits.unwrap(makerTraits) & _ALLOWED_SENDER_MASK);
        return allowedSender == 0 || allowedSender == uint160(sender) & _ALLOWED_SENDER_MASK;
    }

    /**
     * @notice Returns the expiration time of the order.
     * @param makerTraits The traits of the maker.
     * @return result The expiration timestamp of the order.
     */
    function getExpirationTime(MakerTraits makerTraits) internal pure returns (uint256) {
        return (MakerTraits.unwrap(makerTraits) >> _EXPIRATION_OFFSET) & _EXPIRATION_MASK;
    }

    /**
     * @notice Checks if the order has expired.
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the order has expired.
     */
    function isExpired(MakerTraits makerTraits) internal view returns (bool) {
        uint256 expiration = getExpirationTime(makerTraits);
        return expiration != 0 && expiration < block.timestamp;  // solhint-disable-line not-rely-on-time
    }

    /**
     * @notice Returns the nonce or epoch of the order.
     * @param makerTraits The traits of the maker.
     * @return result The nonce or epoch of the order.
     */
    function nonceOrEpoch(MakerTraits makerTraits) internal pure returns (uint256) {
        return (MakerTraits.unwrap(makerTraits) >> _NONCE_OR_EPOCH_OFFSET) & _NONCE_OR_EPOCH_MASK;
    }

    /**
     * @notice Returns the series of the order.
     * @param makerTraits The traits of the maker.
     * @return result The series of the order.
     */
    function series(MakerTraits makerTraits) internal pure returns (uint256) {
        return (MakerTraits.unwrap(makerTraits) >> _SERIES_OFFSET) & _SERIES_MASK;
    }

    /**
      * @notice Determines if the order allows partial fills.
      * @dev If the _NO_PARTIAL_FILLS_FLAG is not set in the makerTraits, then the order allows partial fills.
      * @param makerTraits The traits of the maker, determining their preferences for the order.
      * @return result A boolean indicating whether the maker allows partial fills.
      */
    function allowPartialFills(MakerTraits makerTraits) internal pure returns (bool) {
        return (MakerTraits.unwrap(makerTraits) & _NO_PARTIAL_FILLS_FLAG) == 0;
    }

    /**
     * @notice Checks if the maker needs pre-interaction call.
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the maker needs a pre-interaction call.
     */
    function needPreInteractionCall(MakerTraits makerTraits) internal pure returns (bool) {
        return (MakerTraits.unwrap(makerTraits) & _PRE_INTERACTION_CALL_FLAG) != 0;
    }

    /**
     * @notice Checks if the maker needs post-interaction call.
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the maker needs a post-interaction call.
     */
    function needPostInteractionCall(MakerTraits makerTraits) internal pure returns (bool) {
        return (MakerTraits.unwrap(makerTraits) & _POST_INTERACTION_CALL_FLAG) != 0;
    }

    /**
      * @notice Determines if the order allows multiple fills.
      * @dev If the _ALLOW_MULTIPLE_FILLS_FLAG is set in the makerTraits, then the maker allows multiple fills.
      * @param makerTraits The traits of the maker, determining their preferences for the order.
      * @return result A boolean indicating whether the maker allows multiple fills.
      */
    function allowMultipleFills(MakerTraits makerTraits) internal pure returns (bool) {
        return (MakerTraits.unwrap(makerTraits) & _ALLOW_MULTIPLE_FILLS_FLAG) != 0;
    }

    /**
      * @notice Determines if an order should use the bit invalidator or remaining amount validator.
      * @dev The bit invalidator can be used if the order does not allow partial or multiple fills.
      * @param makerTraits The traits of the maker, determining their preferences for the order.
      * @return result A boolean indicating whether the bit invalidator should be used.
      * True if the order requires the use of the bit invalidator.
      */
    function useBitInvalidator(MakerTraits makerTraits) internal pure returns (bool) {
        return !allowPartialFills(makerTraits) || !allowMultipleFills(makerTraits);
    }

    /**
     * @notice Checks if the maker needs to check the epoch.
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the maker needs to check the epoch manager.
     */
    function needCheckEpochManager(MakerTraits makerTraits) internal pure returns (bool) {
        return (MakerTraits.unwrap(makerTraits) & _NEED_CHECK_EPOCH_MANAGER_FLAG) != 0;
    }

    /**
     * @notice Checks if the maker uses permit2.
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the maker uses permit2.
     */
    function usePermit2(MakerTraits makerTraits) internal pure returns (bool) {
        return MakerTraits.unwrap(makerTraits) & _USE_PERMIT2_FLAG != 0;
    }

    /**
     * @notice Checks if the maker needs to unwraps WETH.
     * @param makerTraits The traits of the maker.
     * @return result A boolean indicating whether the maker needs to unwrap WETH.
     */
    function unwrapWeth(MakerTraits makerTraits) internal pure returns (bool) {
        return MakerTraits.unwrap(makerTraits) & _UNWRAP_WETH_FLAG != 0;
    }
}
