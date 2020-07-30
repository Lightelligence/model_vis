var add = require("./add");
var assert = require("assert");

describe("Running 2 test cases...", function() {
  it("should return correct value for 2+2", function() {
    assert.equal(add(2, 2), 4);
  });
  it("should return correct value for 2+3", function() {
    assert.equal(add(2, 3), 5);
  });
});
