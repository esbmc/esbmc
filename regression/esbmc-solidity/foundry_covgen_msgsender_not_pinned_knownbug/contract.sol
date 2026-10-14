// ===== CityToken.sol =====
pragma solidity >=0.8.0; // solhint-disable-line

/// @title Interface for contracts conforming to ERC-721: Non-Fungible Tokens
/// @author Dieter Shirley <dete@axiomzen.co> (https://github.com/dete)
abstract contract ERC721 {
  // Required methods
  function approve(address _to, uint256 _tokenId) public virtual;
  function balanceOf(address _owner) public view virtual returns (uint256 balance);
  function implementsERC721() public pure virtual returns (bool);
  function ownerOf(uint256 _tokenId) public view virtual returns (address addr);
  function takeOwnership(uint256 _tokenId) public virtual;
  function totalSupply() public view virtual returns (uint256 total);
  function transferFrom(address _from, address _to, uint256 _tokenId) public virtual;
  function transfer(address _to, uint256 _tokenId) public virtual;

  event Transfer(address indexed from, address indexed to, uint256 tokenId);
  event Approval(address indexed owner, address indexed approved, uint256 tokenId);

  // Optional
  // function name() public view returns (string name);
  // function symbol() public view returns (string symbol);
  // function tokenOfOwnerByIndex(address _owner, uint256 _index) external view returns (uint256 tokenId);
  // function tokenMetadata(uint256 _tokenId) public view returns (string infoUrl);
}


contract CityToken is ERC721 {

  /*** EVENTS ***/

  /// @dev The TokenCreated event is fired whenever a new token comes into existence.
  event TokenCreated(uint256 tokenId, string name, uint256 parentId, address owner);

  /// @dev The TokenSold event is fired whenever a token is sold.
  event TokenSold(uint256 tokenId, uint256 oldPrice, uint256 newPrice, address prevOwner, address winner, string name, uint256 parentId);

  /*** CONSTANTS ***/

  /// @notice Name and symbol of the non fungible token, as defined in ERC721.
  string public constant NAME = "CryptoCities"; // solhint-disable-line
  string public constant SYMBOL = "CityToken"; // solhint-disable-line

  uint256 private startingPrice = 0.05 ether;

  /*** STORAGE ***/

  /// @dev A mapping from token IDs to the address that owns them. All tokens have
  ///  some valid owner address.
  mapping (uint256 => address) public tokenIndexToOwner;

  // @dev A mapping from owner address to count of tokens that address owns.
  //  Used internally inside balanceOf() to resolve ownership count.
  mapping (address => uint256) private ownershipTokenCount;

  /// @dev A mapping from TokenIDs to an address that has been approved to call
  ///  transferFrom(). Each Token can only have one approved address for transfer
  ///  at any time. A zero value means no approval is outstanding.
  mapping (uint256 => address) public tokenIndexToApproved;

  // @dev A mapping from TokenIDs to the price of the token.
  mapping (uint256 => uint256) private tokenIndexToPrice;

  // The addresses of the accounts (or contracts) that can execute actions within each roles.
  address public ceoAddress;
  address public cooAddress;

  uint256 private tokenCreatedCount;

  /*** DATATYPES ***/

  struct Token {
    string name;
    uint256 parentId;
  }

  Token[] private tokens;

  mapping(uint256 => Token) private tokenIndexToToken;

  /*** ACCESS MODIFIERS ***/
  /// @dev Access modifier for CEO-only functionality
  modifier onlyCEO() {
    require(msg.sender == ceoAddress);
    _;
  }

  /// @dev Access modifier for COO-only functionality
  modifier onlyCOO() {
    require(msg.sender == cooAddress);
    _;
  }

  /// Access modifier for contract owner only functionality
  modifier onlyCLevel() {
    require(
      msg.sender == ceoAddress ||
      msg.sender == cooAddress
    );
    _;
  }

  /*** CONSTRUCTOR ***/
  constructor() {
    ceoAddress = msg.sender;
    cooAddress = msg.sender;
  }

  /*** PUBLIC FUNCTIONS ***/
  /// @notice Grant another address the right to transfer token via takeOwnership() and transferFrom().
  /// @param _to The address to be granted transfer approval. Pass address(0) to
  ///  clear all approvals.
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function approve(
    address _to,
    uint256 _tokenId
  ) public override {
    // Caller must own token.
    require(_owns(msg.sender, _tokenId));

    tokenIndexToApproved[_tokenId] = _to;

    emit Approval(msg.sender, _to, _tokenId);
  }

  /// For querying balance of a particular account
  /// @param _owner The address for balance query
  /// @dev Required for ERC-721 compliance.
  function balanceOf(address _owner) public view override returns (uint256 balance) {
    return ownershipTokenCount[_owner];
  }

  /// @dev Creates a new Token with the given name, parentId and price and assigns it to an address.
  function createToken(uint256 _tokenId, address _owner, string memory _name, uint256 _parentId, uint256 _price) public onlyCOO {

    address tokenOwner = _owner;
    if (tokenOwner == address(0)) {
      tokenOwner = cooAddress;
    }
    
    if (_price <= 0) {
      _price = startingPrice;
    }

    tokenCreatedCount++;
    _createToken(_tokenId, _name, _parentId, tokenOwner, _price);
  }


  /// @notice Returns all the relevant information about a specific token.
  /// @param _tokenId The tokenId of the token of interest.
  function getToken(uint256 _tokenId) public view returns (
    string memory tokenName,
    uint256 parentId,
    uint256 sellingPrice,
    address owner
  ) {
    Token storage token = tokenIndexToToken[_tokenId];

    tokenName = token.name;
    parentId = token.parentId;
    sellingPrice = tokenIndexToPrice[_tokenId];
    owner = tokenIndexToOwner[_tokenId];
  }

  function implementsERC721() public pure override returns (bool) {
    return true;
  }

  /// @dev Required for ERC-721 compliance.
  function name() public pure returns (string memory) {
    return NAME;
  }

  /// For querying owner of token
  /// @param _tokenId The tokenID for owner inquiry
  /// @dev Required for ERC-721 compliance.
  function ownerOf(uint256 _tokenId)
    public
    view
    override
    returns (address owner)
  {
    owner = tokenIndexToOwner[_tokenId];
    require(owner != address(0));
  }

  function payout(address _to) public onlyCLevel {
    _payout(_to);
  }

  // Alternate function to withdraw less than total balance
  function withdrawFunds(address _to, uint256 amount) public onlyCLevel {
    _withdrawFunds(_to, amount);
  }
  
  // Allows someone to send ether and obtain the token
  function purchase(uint256 _tokenId) public payable {
    
    // Token IDs above 999 are for countries
    if (_tokenId > 999) {
      _purchaseCountry(_tokenId);
    }else {
      _purchaseCity(_tokenId);
    }

  }

  function priceOf(uint256 _tokenId) public view returns (uint256 price) {
    return tokenIndexToPrice[_tokenId];
  }

  /// @dev Assigns a new address to act as the CEO. Only available to the current CEO.
  /// @param _newCEO The address of the new CEO
  function setCEO(address _newCEO) public onlyCEO {
    require(_newCEO != address(0));

    ceoAddress = _newCEO;
  }

  /// @dev Assigns a new address to act as the COO. Only available to the current COO.
  /// @param _newCOO The address of the new COO
  function setCOO(address _newCOO) public onlyCEO {
    require(_newCOO != address(0));

    cooAddress = _newCOO;
  }

  /// @dev Required for ERC-721 compliance.
  function symbol() public pure returns (string memory) {
    return SYMBOL;
  }

  /// @notice Allow pre-approved user to take ownership of a token
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function takeOwnership(uint256 _tokenId) public override {
    address newOwner = msg.sender;
    address oldOwner = tokenIndexToOwner[_tokenId];

    // Safety check to prevent against an unexpected 0x0 default.
    require(_addressNotNull(newOwner));

    // Making sure transfer is approved
    require(_approved(newOwner, _tokenId));

    _transfer(oldOwner, newOwner, _tokenId);
  }

  /// @param _owner The owner whose city tokens we are interested in.
  /// @dev This method MUST NEVER be called by smart contract code. First, it's fairly
  ///  expensive (it walks the entire Cities array looking for cities belonging to owner),
  ///  but it also returns a dynamic array, which is only supported for web3 calls, and
  ///  not contract-to-contract calls.
  function tokensOfOwner(address _owner) public view returns(uint256[] memory ownerTokens) {
    uint256 tokenCount = balanceOf(_owner);
    if (tokenCount == 0) {
        // Return an empty array
      return new uint256[](0);
    } else {
      uint256[] memory result = new uint256[](tokenCount);
      uint256 totalTokens = totalSupply();
      uint256 resultIndex = 0;

      uint256 tokenId;
      for (tokenId = 0; tokenId <= totalTokens; tokenId++) {
        if (tokenIndexToOwner[tokenId] == _owner) {
          result[resultIndex] = tokenId;
          resultIndex++;
        }
      }
      return result;
    }
  }

  /// For querying totalSupply of token
  /// @dev Required for ERC-721 compliance.
  function totalSupply() public view override returns (uint256 total) {
    //return tokens.length;
    // NOTE: Looks like we can't get the length of mapping data structure
    //return tokenIndexToToken.length;
    return tokenCreatedCount;
  }

  /// Owner initates the transfer of the token to another account
  /// @param _to The address for the token to be transferred to.
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function transfer(
    address _to,
    uint256 _tokenId
  ) public override {
    require(_owns(msg.sender, _tokenId));
    require(_addressNotNull(_to));

    _transfer(msg.sender, _to, _tokenId);
  }

  /// Third-party initiates transfer of token from address _from to address _to
  /// @param _from The address for the token to be transferred from.
  /// @param _to The address for the token to be transferred to.
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function transferFrom(
    address _from,
    address _to,
    uint256 _tokenId
  ) public override {
    require(_owns(_from, _tokenId));
    require(_approved(_to, _tokenId));
    require(_addressNotNull(_to));

    _transfer(_from, _to, _tokenId);
  }

  /*** PRIVATE FUNCTIONS ***/

  function _purchaseCity(uint256 _tokenId) private {

     address oldOwner = tokenIndexToOwner[_tokenId];

    // Using msg.sender instead of creating a new var because we have too many vars
    //address newOwner = msg.sender;

    uint256 sellingPrice = tokenIndexToPrice[_tokenId];

    // Making sure token owner is not sending to self
    require(oldOwner != msg.sender);

    // Safety check to prevent against an unexpected 0x0 default.
    require(_addressNotNull(msg.sender));

    // Making sure sent amount is greater than or equal to the sellingPrice
    require(msg.value >= sellingPrice);

    // Payment to previous owner should be 92% of sellingPrice
    // The other 8% is the 6% dev fee (stays in contract) and 2% Country dividend (goes to Country owner)
    // If Country does not exist yet then we add that 2% to what the previous owner gets
    // Formula: sellingPrice * 92 / 100
    // Same as: sellingPrice * .92 / 1
    uint256 payment = uint256(SafeMath.div(SafeMath.mul(sellingPrice, 92), 100));

    // Get parentId of token
    uint256 parentId = tokenIndexToToken[_tokenId].parentId;

    // Get owner address of parent
    address ownerOfParent = tokenIndexToOwner[parentId];

    // Calculate 2% of selling price
    uint256 paymentToOwnerOfParent = uint256(SafeMath.div(SafeMath.mul(sellingPrice, 2), 100));

    // If we have an address for parentId
    // If not that means parent hasn't been created yet
    // For example the city may exist but we haven't created its country yet
    // Parent ID must also be bigger than the 0-999 range of city ids ...
    // ... since a city can't be a parent of another city
    if (_addressNotNull(ownerOfParent)) {

      // Send 2% dividends to owner of parent
      payable(ownerOfParent).transfer(paymentToOwnerOfParent);

    } else {

      // If no parent owner then update payment to previous owner to include paymentToOwnerOfParent
      payment = SafeMath.add(payment, paymentToOwnerOfParent);
     
    }

    // Get amount over purchase price they paid so that we can send it back to them
    uint256 purchaseExcess = SafeMath.sub(msg.value, sellingPrice);

    // Update price so that when 8% is taken out (dev fee + Country dividend) ...
    // ... the owner gets 20% over their investment
    tokenIndexToPrice[_tokenId] = SafeMath.div(SafeMath.mul(sellingPrice, 120), 92);
    
    _transfer(oldOwner, msg.sender, _tokenId);

    // Pay previous tokenOwner if owner is not contract
    if (oldOwner != address(this)) {
      payable(oldOwner).transfer(payment);
    }

    emit TokenSold(_tokenId, sellingPrice, tokenIndexToPrice[_tokenId], oldOwner, msg.sender, tokenIndexToToken[_tokenId].name, parentId);

    payable(msg.sender).transfer(purchaseExcess);
  }

  function _purchaseCountry(uint256 _tokenId) private {

    address oldOwner = tokenIndexToOwner[_tokenId];

    uint256 sellingPrice = tokenIndexToPrice[_tokenId];

    // Making sure token owner is not sending to self
    require(oldOwner != msg.sender);

    // Safety check to prevent against an unexpected 0x0 default.
    require(_addressNotNull(msg.sender));

    // Making sure sent amount is greater than or equal to the sellingPrice
    require(msg.value >= sellingPrice);

    // Payment to previous owner should be 96% of sellingPrice
    // The other 4% is the dev fee (stays in contract) 
    // Formula: sellingPrice * 96 / 10
    // Same as: sellingPrice * .96 / 1
    uint256 payment = uint256(SafeMath.div(SafeMath.mul(sellingPrice, 96), 100));

    // Get amount over purchase price they paid so that we can send it back to them
    uint256 purchaseExcess = SafeMath.sub(msg.value, sellingPrice);

    // Update price so that when 4% is taken out (dev fee) ...
    // ... the owner gets 15% over their investment
    tokenIndexToPrice[_tokenId] = SafeMath.div(SafeMath.mul(sellingPrice, 115), 96);
    
    _transfer(oldOwner, msg.sender, _tokenId);

    // Pay previous tokenOwner if owner is not contract
    if (oldOwner != address(this)) {
      payable(oldOwner).transfer(payment);
    }

    emit TokenSold(_tokenId, sellingPrice, tokenIndexToPrice[_tokenId], oldOwner, msg.sender, tokenIndexToToken[_tokenId].name, 0);

    payable(msg.sender).transfer(purchaseExcess);
  }


  /// Safety check on _to address to prevent against an unexpected 0x0 default.
  function _addressNotNull(address _to) private pure returns (bool) {
    return _to != address(0);
  }

  /// For checking approval of transfer for address _to
  function _approved(address _to, uint256 _tokenId) private view returns (bool) {
    return tokenIndexToApproved[_tokenId] == _to;
  }


  /// For creating City
  function _createToken(uint256 _tokenId, string memory _name, uint256 _parentId, address _owner, uint256 _price) private {
    
    Token memory _token = Token({
      name: _name,
      parentId: _parentId
    });

    // Rather than increment we need to be able to pass in any tokenId
    // Necessary if we are going to decide on parentIds ahead of time when creating cities ...
    // ... and then creating the parent tokens (countries) later
    //uint256 newTokenId = tokens.push(_token) - 1;
    uint256 newTokenId = _tokenId;
    tokenIndexToToken[newTokenId] = _token;

    // NOTE: Now that we don't autoincrement tokenId should we ...
    // ... check to make sure passed _tokenId arg doesn't already exist?
    
    // It's probably never going to happen, 4 billion tokens are A LOT, but
    // let's just be 100% sure we never let this happen.
    require(newTokenId == uint256(uint32(newTokenId)));

    emit TokenCreated(newTokenId, _name, _parentId, _owner);

    tokenIndexToPrice[newTokenId] = _price;

    // This will assign ownership, and also emit the Transfer event as
    // per ERC721 draft
    _transfer(address(0), _owner, newTokenId);
  }

  /// Check for token ownership
  function _owns(address claimant, uint256 _tokenId) private view returns (bool) {
    return claimant == tokenIndexToOwner[_tokenId];
  }

  /// For paying out balance on contract
  function _payout(address _to) private {
    if (_to == address(0)) {
      payable(ceoAddress).transfer(address(this).balance);
    } else {
      payable(_to).transfer(address(this).balance);
    }
  }

  // Alternate function to withdraw less than total balance
  function _withdrawFunds(address _to, uint256 amount) private {
    require(address(this).balance >= amount);
    if (_to == address(0)) {
      payable(ceoAddress).transfer(amount);
    } else {
      payable(_to).transfer(amount);
    }
  }

  /// @dev Assigns ownership of a specific City to an address.
  function _transfer(address _from, address _to, uint256 _tokenId) private {
    // Since the number of cities is capped to 2^32 we can't overflow this
    ownershipTokenCount[_to]++;
    //transfer ownership
    tokenIndexToOwner[_tokenId] = _to;

    // When creating new cities _from is 0x0, but we can't account that address.
    if (_from != address(0)) {
      ownershipTokenCount[_from]--;
      // clear any previously approved ownership exchange
      delete tokenIndexToApproved[_tokenId];
    }

    // Emit the transfer event.
    emit Transfer(_from, _to, _tokenId);
  }
}
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  /**
  * @dev Substracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
// KNOWNBUG: upstream k-induction disables the inductive step here ("cannot
// havoc what a loop writes through a pointer"), so only base cases run and
// the expected proof is never reached.
